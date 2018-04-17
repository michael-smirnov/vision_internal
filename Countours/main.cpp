#include "functions.h"
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    cv::namedWindow("image", CV_WINDOW_NORMAL);
    cv::namedWindow("countours", CV_WINDOW_NORMAL);
	cv::namedWindow("trace", CV_WINDOW_NORMAL);

    cv::Mat img = cv::imread(argv[1]);
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    cv::Mat countours = calc_countours(img_gray, 5);
    cv::Mat crosses = calc_crosses(img_gray);

	cv::imshow("image", img);
	cv::imshow("countours", countours + crosses);

    std::vector<cv::Point> singular_points;

    uint8_t offset = 200;
    for(int i = 0; i < crosses.rows; i++)
    {
        for(int j = 0; j < crosses.cols; j++)
        {
             float col = crosses.at<uint8_t>(i, j);
             if( col > offset )
             {
                 cv::Point p = {j, i};
                 bool should_add = true;
                 for( auto sp : singular_points )
                 {
                     if( std::abs(sp.x - p.x) < 4 && std::abs(sp.y - p.y) < 4)
                     {
                         should_add = false;
                         break;
                     }
                 }

                 if( should_add )
                    singular_points.push_back( {j, i} );
             }
        }
    }

    cv::waitKey();

    std::vector<cv::Vec3i> dir_vectors = {
        { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 },
        { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 }
    };
	cv::Mat trace = cv::Mat::zeros( countours.rows, countours.cols, CV_8U );
    for( auto p : singular_points )
    {
        std::cout << "(" << p.y << "," << p.x << ")" << std::endl;

		cv::Point cur_point = p;
        int last_dir = -1;
		do
		{
			trace.at<uint8_t>(cur_point) = 255;
			cv::imshow("trace", trace);

            cv::Matx33d area;
            cv::Mat roi = countours(cv::Rect{ cur_point.x - 1, cur_point.y - 1, 3, 3 });
            area = roi.clone();
            double directions[8] = {};
            int min_dir = 0;

            min_gradient_direction( area, directions, min_dir );

            if( last_dir >= 0 )
            {
                int left = last_dir > 0 ? last_dir - 1 : 7;
                int right = last_dir < 7 ? last_dir + 1 : 0;

                int min_ind_last = -1;
                const int treshold = 20;

                if( directions[left] < treshold )
                    min_ind_last = left;
                if( directions[last_dir] < treshold && ( min_ind_last == -1 || directions[last_dir] < directions[min_ind_last] ) )
                    min_ind_last = last_dir;
                if( directions[right] < treshold && ( min_ind_last == -1 || directions[right] < directions[min_ind_last] ) )
                    min_ind_last = right;

                if( min_ind_last != -1 )
                    min_dir = min_ind_last;
                else
                {
                    int r = std::abs(min_dir - last_dir);
                    int dist = std::min(r, 8-r);
                    if( dist > 2)
                        break;
                }
            }

            std::cout << "dir:" << dir_vectors[min_dir] << std::endl;

            cur_point.x += dir_vectors[min_dir][0];
            cur_point.y += dir_vectors[min_dir][1];
            last_dir = min_dir;
			
            if( cv::waitKey(100) == 'q' )
                break;
		}
        while (/*cur_point != p*/true);
    }

    cv::waitKey();

    return 0;
}
