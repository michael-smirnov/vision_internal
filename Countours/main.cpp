#include "functions.h"
#include "ComponentMarker.h"
#include "TraceFinder.h"
#include <iostream>
#include <vector>

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
            if( value > 0 )
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

void mouse_callback(int  event, int  x, int  y, int  flag, void *param)
{
    if (event == EVENT_MOUSEMOVE) {
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

    Mat dx_sobel, dy_sobel;
    Sobel(img_gray, dx_sobel, CV_64F, 1, 0);
    Sobel(img_gray, dy_sobel, CV_64F, 0, 1);

    auto angles = calc_angles( dx_sobel, dy_sobel );

    Mat singulars = Mat::zeros( angles.rows, angles.cols, CV_8UC3 );

    int kernel_size = 5;
    int offset = kernel_size / 2;

    for( int i = offset; i < angles.rows - offset; i++ )
    {
        for( int j = offset; j < angles.cols - offset; j++ )
        {
            auto histogram = get_angle_histogram( angles({j-offset, i-offset, kernel_size, kernel_size}) );
            auto peaks = get_massive_histogram_peaks( histogram, 0.1 );

            if( peaks.size() > 3 )
                singulars.at<Vec3b>( i, j ) = { 255, 255, 255 };

            /*int sum = 0;
            int range = 0;
            int last_not_null_element = -1;
            for( int i = 0; i < histogram.size(); i++ )
            {
                sum += histogram[i];
                if( histogram[i] > 0 )
                {
                    if( last_not_null_element >= 0 )
                        range += (i-last_not_null_element);

                    last_not_null_element = i;
                }
            }

            float sum_threshold = static_cast<float>(kernel_size) * kernel_size * 0.5;
            float range_threshold = static_cast<float>(histogram.size()) * 0.8;

            if( sum >= sum_threshold &&
                range >= range_threshold )
            {
                singulars.at<Vec3b>( i, j ) = { 255, 255, 255 };
            }*/
        }
    }

    imshow( wnd_crosses, singulars );
    setMouseCallback( wnd_countours, mouse_callback );
    auto colour_angles = get_colour_angles( angles );

    imshow( wnd_crosses, colour_angles );
    waitKey();
    return 0;
}
