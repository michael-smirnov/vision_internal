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


    auto start_point = get_start_point( countours );
    trace( countours, start_point, tr );

    Mat crosses = calc_crosses( img_gray );
    TraceFinder tf { countours, crosses };

    auto p = tf.next_singular_point();
    while( p.has_value() )
    {
        cout << *p << std::endl;
        p = tf.next_singular_point();
    }

    imshow(wnd_countours, countours);
    imshow(wnd_area, countours({start_point.x-3, start_point.y-3, 7, 7}));
    imshow(wnd_trace, tr);
    imshow(wnd_crosses, crosses);
    waitKey();

    return 0;
}
