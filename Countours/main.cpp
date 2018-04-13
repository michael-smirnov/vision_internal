#include "functions.h"
#include <iostream>

int main()
{
    cv::namedWindow("image", CV_WINDOW_NORMAL);
    cv::namedWindow("countours", CV_WINDOW_NORMAL);
    cv::namedWindow("crosses", CV_WINDOW_NORMAL);

    cv::Mat img = cv::imread("/home/smirnov-ma/internal/vision_internal/Countours/samples/triangle.png");
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    cv::Mat countours = calc_countours(img_gray, 5);
    cv::Mat crosses = calc_crosses(img_gray);

    uint8_t offset = 200;
    for(int i = 0; i < crosses.rows; i++)
    {
        for(int j = 0; j < crosses.cols; j++)
        {
             float col = crosses.at<uint8_t>(i, j);
             if( col > offset )
             {
                 std::cout << "(" << i << "," << j << ") = " << col << std::endl;

                 cv::Point left_down = {j, i};
                 while( left_down.y+1 < crosses.rows &&
                        crosses.at<uint8_t>({left_down.x, left_down.y+1}) > offset )
                 {
                     left_down = {left_down.x, left_down.y+1};
                     while( left_down.x-1 > 0 &&
                            crosses.at<uint8_t>({left_down.x-1, left_down.y}) > offset )
                         left_down = {left_down.x-1, left_down.y};
                 }
             }
        }
    }

    cv::imshow("image", img);
    cv::imshow("countours", countours + crosses);
    cv::imshow("crosses", crosses);

    cv::waitKey();

    return 0;
}
