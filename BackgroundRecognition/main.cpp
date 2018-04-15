#include "AverageCalculator.h"
#include "functions.h"

int main(int argc, char* argv[])
{
    char* video_file = argv[1];
    const int AVERAGE_FRAMES = 60;

	AverageCalculator calculator(AVERAGE_FRAMES);
    cv::VideoCapture capture(video_file);
    cv::Mat frame;

    cv::namedWindow("video", cv::WINDOW_NORMAL);
    cv::namedWindow("contours", cv::WINDOW_NORMAL);
    cv::namedWindow("average", cv::WINDOW_NORMAL);

    cv::waitKey();

	capture >> frame;
	cv::resize(frame, frame, { 640, 480 });
	calculator.add_frame(frame);

	char key = 0;
    while( key != 'q' )
    {
		capture >> frame;

		if (frame.empty())
			break;

		cv::resize(frame, frame, { 640, 480 });
		
		float median_frame = median(frame);
		float median_average = median(calculator.get_average());

		cv::Mat frame_adjusted;
		frame.convertTo(frame_adjusted, CV_32FC3);
		frame_adjusted *= median_average / median_frame;
        cv::medianBlur(frame_adjusted, frame_adjusted, 5);

		calculator.add_frame(frame_adjusted);

		auto average = calculator.get_average();
        auto avg_countours = calc_countours(average, 5);
        auto frame_countours = calc_countours(frame_adjusted, 5);

        int channels[] = { 0 };
        int histSize[] = { 256 };
        float ranges_array[] = { 0.0f, 256.0f };
        const float* ranges = { ranges_array };

        avg_countours *= 10;

        cv::Mat histogram;
        cv::calcHist(&avg_countours, 1, channels, cv::Mat(), histogram, 1, histSize, &ranges );
        draw_histogram( histogram, avg_countours, cv::Scalar(255,255,255) );

        cv::imshow("video", frame);
        cv::imshow("average", frame_countours - avg_countours);
        cv::imshow("contours", avg_countours);

        key = cv::waitKey(1);
    }

    cv::destroyAllWindows();
    return 0;
}
