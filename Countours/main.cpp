#include "functions.h"
#include "ComponentMarker.h"
#include <iostream>
#include <vector>

const std::string wnd_countours = "countours";
const std::string wnd_area = "area";
const std::string wnd_trace = "trace";
const std::string wnd_crosses = "crosses";

cv::Point2i get_first_point( const cv::Mat& img, int offset, int value_threshold )
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

cv::Point2i get_start_point( const cv::Mat& image )
{
    constexpr int value_threshold = 50;
    constexpr int range = 5;
    constexpr int offset = range / 2;

    cv::Mat dx = generate_by_row({-1.0f, -1.0f, 0.0f, 1.0f, 1.0f});
    cv::Mat dy = dx.t();

    auto start_point = get_first_point( image, offset, value_threshold );

    cv::Mat area;
    image({ start_point.x - offset, start_point.y-offset, range, range }).convertTo(area, CV_32F);

    float dx_value = cv::sum(dx.mul(area))[0] / range;
    float dy_value = cv::sum(dy.mul(area))[0] / range;
    cv::Vec2f first_grad = closest_direction(cv::Vec2f{ dx_value, dy_value }, 3);
    cv::Vec2f cur_grad = first_grad;
    cv::Point2i cur_point = start_point;

    while( cv::normalize( cur_grad ).dot( cv::normalize( first_grad) ) > 0.5 )
    {
        cur_point.x += cur_grad[0];
        cur_point.y += cur_grad[1];

        image({ cur_point.x-offset, cur_point.y-offset, range, range }).convertTo(area, CV_32F);
        float dx_value = cv::sum(dx.mul(area))[0] / range;
        float dy_value = cv::sum(dy.mul(area))[0] / range;

        cur_grad = closest_direction(cv::Vec2f{ dx_value, dy_value }, 3);
    }

    return cur_point;
}

void trace( const cv::Mat& img, const cv::Point& start_point, cv::Mat& tr, const cv::Point& last_point = {0, 0} )
{
    double dirs[8] = {};
    cv::Point pts[8] =
    {
        cv::Point{ start_point.x-1, start_point.y-1},
        cv::Point{ start_point.x, start_point.y-1},
        cv::Point{ start_point.x+1, start_point.y-1},
        cv::Point{ start_point.x-1, start_point.y},
        cv::Point{ start_point.x+1, start_point.y},
        cv::Point{ start_point.x-1, start_point.y+1},
        cv::Point{ start_point.x, start_point.y+1},
        cv::Point{ start_point.x+1, start_point.y+1}
    };

    int max_index = 0;

    tr.at<uint8_t>( start_point ) = 255;
    //cv::imshow(wnd_trace, tr);
    //cv::waitKey(10);

    max_gradient_direction( img({start_point.x-1, start_point.y-1, 3, 3}), dirs, max_index );

    for(int i = 0; i < 8; i++)
    {
        if( dirs[i] > 0.70 && pts[i] != last_point && tr.at<uint8_t>( pts[i] ) == 0 )
            trace( img, pts[i], tr, start_point );
    }
}

int main(int argc, char* argv[])
{
    cv::namedWindow(wnd_countours, cv::WINDOW_NORMAL);
    cv::namedWindow(wnd_area, cv::WINDOW_NORMAL);
    cv::namedWindow(wnd_trace, cv::WINDOW_NORMAL);
    cv::namedWindow(wnd_crosses, cv::WINDOW_NORMAL);

    cv::Mat img = cv::imread(argv[1]);
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    cv::Mat countours = calc_countours(img_gray, 3);
    cv::Mat tr = cv::Mat::zeros(countours.rows, countours.cols, CV_8U);
    auto start_point = get_start_point( countours );
    trace( countours, start_point, tr );

    cv::Mat crosses = calc_crosses( img_gray );

    cv::imshow(wnd_countours, countours);
    cv::imshow(wnd_area, countours({start_point.x-3, start_point.y-3, 7, 7}));
    cv::imshow(wnd_trace, tr);
    cv::imshow(wnd_crosses, crosses);
    cv::waitKey();

    return 0;
}
