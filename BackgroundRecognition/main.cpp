#include "AverageCalculator.h"
#include "ImageProcessor.h"
#include "Statistics.h"
#include "Utility.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace vision;

int main(int argc, char* argv[])
{
    char* video_file = argv[1];
    const int AVERAGE_FRAMES = 120;

	AverageCalculator calculator(AVERAGE_FRAMES);
    cv::VideoCapture capture(video_file);
    Mat frame;

    cv::namedWindow("video", cv::WINDOW_NORMAL);
    cv::namedWindow("contours", cv::WINDOW_NORMAL);
    cv::namedWindow("average", cv::WINDOW_NORMAL);

    cv::waitKey();

	capture >> frame;
	cv::resize(frame, frame, { 320, 240 });
	calculator.add_frame(frame);

	char key = 0;
    while( key != 'q' )
    {
		capture >> frame;

		if (frame.empty())
			break;

		cv::resize(frame, frame, { 320, 240 });
		
		float median_frame = Statistics::median(frame);
		float median_average = Statistics::median(calculator.get_average());

		cv::Mat frame_adjusted;
		frame.convertTo(frame_adjusted, CV_32FC3);
        frame_adjusted *= median_average / median_frame;
        cv::medianBlur(frame_adjusted, frame_adjusted, 5);

		calculator.add_frame(frame_adjusted);

		auto average = calculator.get_average();
        auto avg_countours = ImageProcessor::countours( average, 5 );

        int channels[] = { 0 };
        int histSize[] = { 256 };
        float ranges_array[] = { 0.0f, 256.0f };
        const float* ranges = { ranges_array };

        avg_countours *= 10;

        Mat histogram;
        cv::calcHist( &avg_countours, 1, channels, Mat(), histogram, 1, histSize, &ranges );
        Utility::add_histogram( histogram, Scalar(255,255,255), avg_countours );

        cv::imshow("video", frame);
        cv::imshow("average", average);
        cv::imshow("contours", avg_countours);

        key = cv::waitKey(1);
    }

    cv::destroyAllWindows();
    return 0;
}
