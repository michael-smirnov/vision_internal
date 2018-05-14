#include "TraceFinder.h"
#include "functions.h"

using namespace cv;
using namespace std;

TraceFinder::TraceFinder( const Mat& countours , const Mat& singulars )
    : _singular_point_threshold(50)
{
    _tr = Mat::zeros(countours.rows, countours.cols, CV_8U);

    _area3x3_vectors = all_directions( 3 );

    countours.copyTo( _countours );
    _countours.convertTo( _countours, CV_8U );

    singulars.copyTo( _singulars );
    _singulars.convertTo( _singulars, CV_8U );
}

std::vector<vision::trace> TraceFinder::find_traces()
{
    auto sp = next_singular_point();
    while( sp != nullopt )
    {
        auto start_point = max_gradient_singular_point_in_area( sp.value() );
        mark_singular_area( sp.value() );

        for( auto& d : local_directions( start_point ) )
        {
            vision::trace tr;
            tr.start = start_point;
            tr.directions.push_back( _area3x3_vectors[d] );

            traverse( tr );

            if( tr.directions.size() >= 430 )
                _traces.push_back( tr );
        }

        sp = next_singular_point();
    }

    return _traces;
}

optional<Point> TraceFinder::next_singular_point()
{
    for( int i = 0; i < _singulars.rows; i++ )
    {
        for( int j = 0; j < _singulars.cols; j++ )
        {
            Point current = { j, i };
            if( _singulars.at<uint8_t>( current ) > _singular_point_threshold &&
                _spent_singular_points.find( current ) == _spent_singular_points.end() )
            {
                return current;
            }
        }
    }

    return {};
}

size_t TraceFinder::PointHash::operator()( const Point& p ) const
{
    return p.x * 10000 + p.y;
}

Point TraceFinder::max_gradient_singular_point_in_area( const Point& start_singular_point )
{
    auto is_singular = [this]( const Point& point ) { return _singulars.at<uint8_t>( point ) > _singular_point_threshold; };
    auto cos_between_vectors = [this]( const Vec2f& v1, const Vec2f& v2 ) { return normalize(v1).dot( normalize(v2) ); };

    if( !is_singular( start_singular_point ) )
        return { -1, -1 };

    Mat dx = generate_by_row({-1.0f, -1.0f, 0.0f, 1.0f, 1.0f});
    Mat dy = dx.t();

    int degree = dx.rows;
    int offset = degree / 2;

    Mat area;
    _countours( { start_singular_point.x - offset,
                  start_singular_point.y - offset,
                  degree,
                  degree } ).convertTo( area, CV_32F );

    float dx_value = sum(dx.mul(area))[0] / degree;
    float dy_value = sum(dy.mul(area))[0] / degree;

    Vec2f first_grad = closest_direction( Vec2f{ dx_value, dy_value }, 3 );

    Vec2f cur_grad = first_grad;
    float cur_norm = norm( cur_grad );
    Point2i cur_point = start_singular_point;

    while( cos_between_vectors( first_grad, cur_grad ) > 0.7 /*&& cur_norm > 10.0f*/ )
    {
        if( !is_singular( {cur_point.x + cur_grad[0], cur_point.y + cur_grad[1]} ) )
        {
            auto directions = all_directions( 3 );

            int closest_to_cur_dir = -1;
            float max_value = -1.0f;
            for( int i = 0; i < directions.size(); i++ )
            {
                auto& dir = directions[i];

                float cos = cos_between_vectors( first_grad, dir );
                if( cos > 0.7 &&
                    is_singular( {cur_point.x + dir[0], cur_point.y + dir[1]} ) &&
                    cos > max_value )
                {
                    max_value = cos;
                    closest_to_cur_dir = i;
                }
            }

            if( closest_to_cur_dir != -1 )
                cur_grad = directions[ closest_to_cur_dir ];
            else
                return cur_point;
        }

        cur_point.x += cur_grad[0];
        cur_point.y += cur_grad[1];

        _countours({ cur_point.x-offset, cur_point.y-offset, degree, degree }).convertTo(area, CV_32F);

        float dx_value = sum(dx.mul(area))[0] / degree;
        float dy_value = sum(dy.mul(area))[0] / degree;

        cur_grad = closest_direction(Vec2f{ dx_value, dy_value }, 3);
    }

    return cur_point;
}

vector<int> TraceFinder::local_directions( const Point& origin )
{
    constexpr int dirs_count = 8;
    double dirs[ dirs_count ] = {};
    int max_index = -1;

    max_gradient_direction( _countours({origin.x-1, origin.y-1, 3, 3}), dirs, max_index );
    std::vector<int> result = {};

    for( int i = 0; i < dirs_count; i++ )
    {
        auto& v = _area3x3_vectors[i];
        if( dirs[i] > 0.7 &&
            _spent_regular_points.find( { origin.x + v[0], origin.y + v[1] } ) == _spent_regular_points.end() )
        {
            result.push_back(i);
        }
    }

    return result;
}

int TraceFinder::global_direction( const Point& origin, const Vec2f direction )
{
    constexpr int dirs_count = 8;
    double dirs[ dirs_count ] = {};
    int max_index = -1;

    max_gradient_direction( _countours({origin.x-1, origin.y-1, 3, 3}), dirs, max_index );

    int best_direction = -1;
    double best_value = 0.8;

    for( int i = 0; i < dirs_count; i++ )
    {
        auto& v = _area3x3_vectors[i];

        double direction_dot = normalize( direction ).dot( normalize(v) );
        double direction_value = dirs[i];
        double cur_value = (0.6*direction_dot + 0.4*direction_value);

        if( cur_value > best_value &&
            _spent_regular_points.find( { origin.x + v[0], origin.y + v[1] } ) == _spent_regular_points.end() )
        {
            best_direction = i;
            best_value = cur_value;
        }
    }

    return best_direction;
}

void TraceFinder::traverse( vision::trace& t )
{
    Point last_point = get_last_point( t );

    /*_tr.at<uint8_t>( last_point ) = 255;
    _ar = _countours( { last_point.x - 2, last_point.y - 2, 5, 5 } );
    cv::Mat cr = _singulars( { last_point.x - 2, last_point.y - 2, 5, 5 } );

    imshow( "crosses", cr );
    imshow( "trace", _tr );
    imshow( "area", _ar );
    cv::waitKey();*/

    if( _singulars.at<uint8_t>( last_point ) > _singular_point_threshold &&
         _spent_singular_points.find( last_point ) == _spent_singular_points.end() )
    {
        return;
    }
    else
    {
        int dir = global_direction( last_point, t.directions.back() );
        if( dir != -1 )
        {
            t.directions.push_back( _area3x3_vectors[ dir ] );
            traverse( t );
        }
    }
}

Point TraceFinder::get_last_point( vision::trace& t )
{
    Point last_point = t.start;

    for( int i = 0; i < t.directions.size(); i++ )
    {
        auto& dir = t.directions[i];

        last_point.x += dir[0];
        last_point.y += dir[1];
    }
    return last_point;
}

void TraceFinder::mark_singular_area( const Point& origin )
{
    if( _singulars.at<uint8_t>( origin ) > _singular_point_threshold &&
        _spent_singular_points.find( origin ) == _spent_singular_points.end() )
    {
        _spent_singular_points.insert( origin );

        for( auto& dir : _area3x3_vectors )
        {
            mark_singular_area( { origin.x + dir[0], origin.y + dir[1] } );
        }
    }
}
