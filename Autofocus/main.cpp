#include "MathItemGenerator.h"
#include "ImageProcessor.h"
#include "Statistics.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace vision;
using namespace cv;

bool read_frame( Mat& src, VideoCapture& capture);
void mouse_callback( int event, int x, int y, int flags, void* userdata );

int main(int argc, char* argv[])
{
    char* video_file = argv[1];

	Mat frame;
	Mat frame_area;

	VideoCapture capture(video_file);
	Rect area;

	namedWindow("window", WINDOW_NORMAL);
    namedWindow("area", WINDOW_NORMAL);
    namedWindow("autofocus", WINDOW_NORMAL);
    namedWindow("lines", WINDOW_NORMAL);

	setMouseCallback("window", mouse_callback, &area);

	read_frame(frame, capture);

	char key = 0;
    while (key != 13)
	{
		Mat tmp = frame.clone();
		rectangle(tmp, area, Scalar(255, 255, 255));
		imshow("window", tmp);
        key = waitKey(33);
	}

	waitKey();
    bool first_show = true;
    float autofocus_value = 0.0f;

	auto horizontal_filter = MathItemGenerator::matrix_by_row( { 0.0f, 1.0f, 4.0f, -1.0f, -8.0f, -1.0f, 4.0f, 1.0f, 0.0f } );
	auto vertical_filter = horizontal_filter.t();

	while (key != 'q')
	{
		if (read_frame(frame, capture))
		{
            Mat crosses = ImageProcessor::crosses( frame, horizontal_filter, vertical_filter );

            frame_area = frame(area);
            Mat countours = ImageProcessor::countours( frame_area, 5 );
			{
				Mat hist;
				int channels[] = { 0 };
				int histSize[] = { 256 };
				float ranges[] = { 0.0f, 256.0f };
				const float* ranges_arr[] = { ranges };

				calcHist(&countours, 1, channels, Mat(), hist, 1, histSize, ranges_arr);

                float max, dev;
				Statistics::local_max_and_stddev( (float*)hist.data, histSize[0], 0.01f, max, dev );
                float new_autofocus_value = max / dev;

                if ( first_show || autofocus_value > new_autofocus_value )
                {
                    first_show = false;
                    autofocus_value = new_autofocus_value;
                    imshow("autofocus", frame);
                }
			}
            medianBlur(countours, countours, 5);
            medianBlur(countours, countours, 5);
            countours.convertTo(countours, CV_8U);

			imshow("window", frame);
            imshow("area", countours);
            imshow("lines", crosses);
		}

		key = waitKey(33);
        if ( key == 'w')
            key = waitKey();
	}

    destroyAllWindows();
    return 0;
}

bool read_frame(Mat& src, VideoCapture& capture)
{
    capture >> src;

    if (src.empty())
        return false;

    cvtColor(src, src, COLOR_BGR2GRAY);

    return true;
}

void mouse_callback(int event, int x, int y, int flags, void* userdata)
{
	static bool selection_started = false;
	Rect* r = (Rect*)userdata;

	if (event == EVENT_LBUTTONDOWN)
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
	else if (event == EVENT_MOUSEMOVE && selection_started)
	{
		r->width = x - r->x;
		r->height = y - r->y;
	}
}
