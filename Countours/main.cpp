#include "functions.h"
#include "ComponentMarker.h"
#include "TraceFinder.h"

#include <iostream>
#include <vector>
#include <exception>

using namespace std;
using namespace cv;

const string wnd_countours = "countours";
const string wnd_area = "area";
const string wnd_trace = "trace";
const string wnd_crosses = "crosses";

Point2i get_first_point( const Mat& img, int offset, int value_threshold )
{
    for(int i = offset; i < img.rows - offset; i++)
    {
        for(int j = offset; j < img.cols - offset; j++)
        {
            uchar element = img.at<uchar>(i, j);
            if( element > value_threshold )
                return {j, i};
        }
    }

    return {-1, -1};
}

Point2i get_start_point( const Mat& image )
{
    constexpr int value_threshold = 50;
    constexpr int range = 5;
    constexpr int offset = range / 2;

    Mat dx = generate_by_row({-1.0f, -1.0f, 0.0f, 1.0f, 1.0f});
    Mat dy = dx.t();

    auto start_point = get_first_point( image, offset, value_threshold );

    Mat area;
    image({ start_point.x - offset, start_point.y-offset, range, range }).convertTo(area, CV_32F);

    float dx_value = sum(dx.mul(area))[0] / range;
    float dy_value = sum(dy.mul(area))[0] / range;
    Vec2f first_grad = closest_direction(Vec2f{ dx_value, dy_value }, 3);
    Vec2f cur_grad = first_grad;
    Point2i cur_point = start_point;

    while( normalize( cur_grad ).dot( normalize( first_grad ) ) > 0.5 )
    {
        cur_point.x += cur_grad[0];
        cur_point.y += cur_grad[1];

        image({ cur_point.x-offset, cur_point.y-offset, range, range }).convertTo(area, CV_32F);
        float dx_value = sum(dx.mul(area))[0] / range;
        float dy_value = sum(dy.mul(area))[0] / range;

        cur_grad = closest_direction(Vec2f{ dx_value, dy_value }, 3);
    }

    return cur_point;
}

void trace( const Mat& img, const Point& start_point, Mat& tr, const Point& last_point = {0, 0} )
{
    double dirs[8] = {};
    Point pts[8] =
    {
        Point{ start_point.x-1, start_point.y-1},
        Point{ start_point.x, start_point.y-1},
        Point{ start_point.x+1, start_point.y-1},
        Point{ start_point.x-1, start_point.y},
        Point{ start_point.x+1, start_point.y},
        Point{ start_point.x-1, start_point.y+1},
        Point{ start_point.x, start_point.y+1},
        Point{ start_point.x+1, start_point.y+1}
    };

    int max_index = 0;

    tr.at<uint8_t>( start_point ) = 255;

    max_gradient_direction( img({start_point.x-1, start_point.y-1, 3, 3}), dirs, max_index );

    for(int i = 0; i < 8; i++)
    {
        if( dirs[i] > 0.70 && pts[i] != last_point && tr.at<uint8_t>( pts[i] ) == 0 )
            trace( img, pts[i], tr, start_point );
    }
}

std::vector<int> get_angle_histogram( const cv::Mat& m, uint32_t histogram_length = 36 )
{
    vector<int> v( histogram_length, 0 );

    for( int i = 0; i < m.rows; i++ )
    {
        for( int j = 0; j < m.cols; j++ )
        {
            auto value = m.at<int16_t>( i, j );
            if( value >= 0 )
            {
                float normalized_value = static_cast<float>(value) / 360.0f;
                int bin = normalized_value * (histogram_length-1);
                v[ bin ]++;
            }
        }
    }

    return v;
}

std::vector<int> get_massive_histogram_peaks( std::vector<int> histogram, float threshold_persent = 0.1 )
{
    int sum = 0;
    for( auto& element : histogram )
        sum += element;

    std::vector<int> peaks;
    int current_peak_sum = 0;
    for( int i = 0; i < histogram.size(); i++ )
    {
        if( histogram[i] > 0 )
        {
            current_peak_sum += histogram[i];
        }
        else if( current_peak_sum > 0 )
        {
            if( current_peak_sum > sum * threshold_persent )
                peaks.push_back( current_peak_sum );
            current_peak_sum = 0;
        }
    }

    return peaks;
}

int get_median_index_only_by_increment( std::vector<int> histogram, int half_window, int start_index )
{
    int cur_index = start_index;

    int sum = histogram[cur_index];
    for( int j = cur_index + 1; j < histogram.size() && j < cur_index + 1 + half_window; j++ )
        sum += histogram[j];

    int k = histogram[cur_index];
    while( k < sum / 2 )
    {
        cur_index++;
        k += histogram[cur_index];
    }

    return cur_index;
}

int get_median_index_by_center( std::vector<int> histogram, int half_window, int center_index )
{
    int last_histogram_index = histogram.size() - 1;

    int start_index = max( 0, center_index - half_window );
    int end_index = min( last_histogram_index, center_index + half_window );

    int sum = 0;
    for( int i = start_index; i <= end_index; i++ )
        sum += histogram[i];

    if( sum == 1 )
        return center_index;

    int cur_index = start_index;

    int k = histogram[cur_index];
    while( k < sum / 2 )
    {
        cur_index++;
        k += histogram[cur_index];
    }

    return cur_index;
}

std::vector<int> get_most_valuable_arguments( std::vector<int> histogram, int half_window )
{
    std::vector<int> args;

    int i = 0;
    while( i < histogram.size() )
    {
        if( histogram[i] == 0 )
            i++;
        else
        {
            i = get_median_index_only_by_increment( histogram, half_window, i );
            i = get_median_index_by_center( histogram, half_window, i );

            args.push_back( i );
            i += half_window + 1;
        }
    }

    return args;
}

float get_probability_of_center_is_end_point( const Mat& m, double value_percent = 0.9 )
{
    if( m.rows != m.cols )
        throw runtime_error( "nonsquare matrix" );

    if( m.type() != CV_64F )
        throw runtime_error( "invalid type of matrix" );

    std::vector<Vec2i> dir =
    {
        { -1, -1 },
        {  0, -1 },
        {  1, -1 },
        { -1,  0 },
        {  1,  0 },
        { -1,  1 },
        {  0,  1 },
        {  1,  1 },
    };

    std::vector<int> max_distance( dir.size(), -1 );

    int distance_range = m.rows / 2;
    double center_element = m.at<double>( distance_range, distance_range );

    int count_of_not_initialized_values = 0;
    for( int i = 0; i < dir.size(); i++ )
    {
        for( int j = 1; j <= distance_range; j++ )
        {
            double current_element = m.at<double>( distance_range + dir[i][1], distance_range + dir[i][0] );
            if( current_element / center_element < value_percent )
            {
                max_distance[i] = j-1;
                break;
            }
        }

        if( max_distance[i] < 0 )
            count_of_not_initialized_values++;
    }

    return /*1.0 - static_cast<double>(count_of_not_initialized_values) / dir.size()*/ count_of_not_initialized_values == 7 ? 1.0f : 0.0f;
}

void mouse_callback( int event, int x, int y, int flag, void* param )
{
    if (event == EVENT_MOUSEMOVE)
    {
        cout << "(" << x << ", " << y << ")" << endl;
    }
}

int main(int argc, char* argv[])
{
    namedWindow(wnd_countours, WINDOW_NORMAL);
    namedWindow(wnd_area, WINDOW_NORMAL);
    namedWindow(wnd_trace, WINDOW_NORMAL);
    namedWindow(wnd_crosses, WINDOW_NORMAL);

    Mat img = imread(argv[1]);
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

    Mat countours = calc_countours(img_gray, 3);
    Mat tr = Mat::zeros(countours.rows, countours.cols, CV_8U);
    Mat crosses = calc_crosses( img_gray );

    Mat dx_sobel, dy_sobel, mag_sobel;
    Sobel(img_gray, dx_sobel, CV_64F, 1, 0);
    Sobel(img_gray, dy_sobel, CV_64F, 0, 1);

    cv::sqrt( dx_sobel.mul(dx_sobel) + dy_sobel.mul(dy_sobel), mag_sobel );
    Mat img_magnitude;
    mag_sobel.convertTo( img_magnitude, CV_8UC3 );

    auto angles = calc_angles( dx_sobel, dy_sobel );
    auto colour_angles = get_colour_angles( angles );

    Mat singulars = Mat::zeros( angles.rows, angles.cols, CV_8UC3 );

    int kernel_size = 5;
    int offset = kernel_size / 2;

    /*for( int i = offset; i < angles.rows - offset; i++ )
    {
        for( int j = offset; j < angles.cols - offset; j++ )
        {
            auto histogram = get_angle_histogram( angles({j-offset, i-offset, kernel_size, kernel_size}) );
            auto peaks = get_most_valuable_arguments( histogram, 5 );

            if( peaks.size() == 2 )
                singulars.at<Vec3b>( i, j ) = { 100, 100, 100 };
            else if( peaks.size() == 3 )
                singulars.at<Vec3b>( i, j ) = { 180, 180, 180 };
            else if( peaks.size() > 3 )
                singulars.at<Vec3b>( i, j ) = { 255, 255, 255 };
        }
    }*/

    for( int i = offset; i < mag_sobel.rows - offset; i++ )
    {
        for( int j = offset; j < mag_sobel.cols - offset; j++ )
        {
            float p = get_probability_of_center_is_end_point( mag_sobel({j-offset, i-offset, kernel_size, kernel_size}) );
            if( p > 0.5 )
            {
                imshow( wnd_area, img_magnitude({j-offset, i-offset, kernel_size, kernel_size}) );
                waitKey();
            }
            singulars.at<Vec3b>( i, j ) = { 255 * p, 255 * p, 255 * p };
        }
    }

    setMouseCallback( wnd_countours, mouse_callback );

    imshow( wnd_countours, colour_angles );
    imshow( wnd_crosses, singulars + colour_angles );

    waitKey();
    return 0;
}
