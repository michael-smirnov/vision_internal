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

	cv::Mat grad_vert = generate_by_row({ -1.0f, -1.0f, 0.0f, 1.0f, 1.0f });
	cv::Mat grad_hori = grad_vert.t();

	//cv::namedWindow("grad", CV_WINDOW_NORMAL);

	std::vector<cv::Vec2d> directions = 
	{
		cv::Vec2d{ -1.0, -1.0 },
		cv::Vec2d{ 0.0, -1.0 },
		cv::Vec2d{ 1.0, -1.0 },
		cv::Vec2d{ 1.0, 0.0 },
		cv::Vec2d{ 1.0, 1.0 },
		cv::Vec2d{ 0.0, 1.0 },
		cv::Vec2d{ -1.0, 1.0 },
		cv::Vec2d{ -1.0, 0.0 }
	};

	cv::Mat trace = cv::Mat::zeros( countours.rows, countours.cols, CV_8U );
    for( auto p : singular_points )
    {
        std::cout << "(" << p.y << "," << p.x << ")" << std::endl;

		cv::Point cur_point = p;

		do
		{
			trace.at<uint8_t>(cur_point) = 255;
			cv::imshow("trace", trace);

			cv::Mat area = countours(cv::Rect{ cur_point.x - grad_vert.cols / 2, cur_point.y - grad_vert.rows / 2, grad_vert.cols, grad_vert.rows }).clone();
			/*cv::imshow("grad", area);
			cv::rectangle(img, cv::Rect{ p.x - grad_vert.cols / 2, p.y - grad_vert.rows / 2, grad_vert.cols, grad_vert.rows }, cv::Scalar(255, 0, 0));
			cv::imshow("image", img);*/
			area.convertTo(area, CV_32F);

			auto mdx = area.mul(grad_vert);
			auto mdy = area.mul(grad_hori);

			auto dx = (cv::sum(mdx) / grad_vert.cols)[0];
			auto dy = (cv::sum(mdy) / grad_vert.rows)[0];

			cv::Vec2d grad = { dx, dy };
			grad = cv::normalize(grad);
			cv::Vec2d dir = { grad[1], -grad[0] };

			std::cout << "grad:" << grad << "dir:" << dir << std::endl;

			std::vector<int> possible_directions;
			int best_direction = -1;
			int best_value = 255;

			for (int i = 0; i < directions.size(); i++)
			{
				cv::Vec2d curdir = cv::normalize(directions[i]);
				if (curdir.dot(dir) > 0.7 || curdir.dot(dir) < -0.7)
				{
					possible_directions.push_back(i);
					int x = cur_point.x + directions[i][0];
					int y = cur_point.y + directions[i][1];

					int v_new = countours.at<uint8_t>( {x, y} );
					int v_old = countours.at<uint8_t>( cur_point );
					int dv = std::abs(v_new - v_old);
					if ( dv < best_value)
					{
						best_value = dv;
						best_direction = i;
					}
				}
			}

			if (best_direction == -1)
				break;

			cur_point.x += directions[best_direction][0];
			cur_point.y += directions[best_direction][1];
			
			cv::waitKey();
		}
		while (cur_point != p);
    }

    cv::waitKey();

    return 0;
}
