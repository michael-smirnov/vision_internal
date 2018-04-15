#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "functions.h"

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	static bool selection_started = false;
	cv::Rect* r = (cv::Rect*)userdata;

	if (event == cv::EVENT_LBUTTONDOWN)
	{
		if (!selection_started)
		{
			selection_started = true;
			r->x = x;
			r->y = y;
		}
		else
			selection_started = false;
	}
	else if (event == cv::EVENT_MOUSEMOVE && selection_started)
	{
		r->width = x - r->x;
		r->height = y - r->y;
	}
}

int main(int argc, char* argv[])
{
    char* video_file = argv[1];

	cv::VideoCapture capture(video_file);
	cv::Mat frame;
	cv::Mat frame_area;
	cv::Rect area;

	cv::namedWindow("window", cv::WINDOW_NORMAL);
    cv::namedWindow("area", cv::WINDOW_NORMAL);
    cv::namedWindow("autofocus", cv::WINDOW_NORMAL);
    cv::namedWindow("lines", cv::WINDOW_NORMAL);

	cv::setMouseCallback("window", CallBackFunc, &area);

	read_frame(frame, capture);

	char key = 0;
    while (key != 10)
	{
		cv::Mat tmp = frame.clone();
		cv::rectangle(tmp, area, cv::Scalar(255, 255, 255));
		cv::imshow("window", tmp);
        key = cv::waitKey(33);
	}

	cv::waitKey();
    bool first_show = true;
    float autofocus_value = 0.0f;

	while (key != 'q')
	{
		if (read_frame(frame, capture))
		{
            cv::Mat crosses = calc_crosses(frame);

            frame_area = frame(area);
            cv::Mat countours = calc_countours(frame_area, 5);
			{
				cv::Mat hist;
				int channels[] = { 0 };
				int histSize[] = { 256 };
				float ranges[] = { 0.0f, 256.0f };
				const float* ranges_arr[] = { ranges };

				cv::calcHist(&countours, 1, channels, cv::Mat(), hist, 1, histSize, ranges_arr);

                float max, dev;
                calc_local_max_deviation( (float*)hist.data, histSize[0], 0.01f, max, dev );
                float new_autofocus_value = max / dev;

                //draw_histogram(hist, countours, cv::Scalar(255, 255, 255));

                if ( first_show || autofocus_value > new_autofocus_value )
                {
                    first_show = false;
                    autofocus_value = new_autofocus_value;
                    cv::imshow("autofocus", frame);
                }
			}
            cv::medianBlur(countours, countours, 5);
            cv::medianBlur(countours, countours, 5);
            countours.convertTo(countours, CV_8U);

			cv::imshow("window", frame);
            //cv::imshow("area", countours);
            cv::imshow("lines", crosses);
		}

		key = cv::waitKey(33);
        if ( key == 'w')
            key = cv::waitKey();
	}

    cv::destroyAllWindows();
    return 0;
}
