#include "functions.h"

void draw_histogram(const cv::Mat& hist, const cv::Mat& img, const cv::Scalar& color)
{
	int hist_w = 256;
	int hist_h = img.rows - img.rows * 0.01;
	int bin_w = cvRound((double)hist_w / 256);

	cv::normalize(hist, hist, 0, img.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 1
         ; i < 256; i++)
		cv::line(img,
			cv::Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			color,
			2,
			cv::LINE_8,
			0);
}

float median(const cv::Mat& m)
{
	int channels[] = { 0 };
	int histSize[] = { 256 };
	float ranges_array[] = { 0.0f, 256.0f };
	const float* ranges = { ranges_array };

	cv::Mat histogram;
	cv::calcHist(&m, 1, channels, cv::Mat(), histogram, 1, histSize, &ranges);

	float median = -1.0f;
	int in_bins = 0;
	float middle = (m.rows * m.cols) / 2.0f;
	for (int i = 0; i < *histSize; i++)
	{
		in_bins += cvRound(histogram.at<float>(i));
		if (in_bins >= middle)
		{
			median = i;
			break;
		}
	}

	return median;
}

void mean_stddev(const cv::Mat& src, cv::Mat& mean, cv::Mat& stddev, int window_size)
{
	float n = window_size * window_size;

	cv::Mat kernel = cv::Mat::zeros(window_size + 1, window_size + 1, CV_32F);
	kernel.at<float>(0, 0) = 1.0f / n;
	kernel.at<float>(0, window_size) = -1.0f / n;
	kernel.at<float>(window_size, 0) = -1.0f / n;
	kernel.at<float>(window_size, window_size) = 1.0f / n;

	cv::Mat sum;
	cv::Mat sqsum;

	cv::integral(src, sum, sqsum);
	sum.convertTo(sum, CV_32FC3);

	cv::filter2D(sum, mean, -1, kernel);
	cv::filter2D(sqsum, stddev, -1, kernel);

	mean.convertTo(mean, sqsum.type());
	stddev = stddev - mean.mul(mean);
	cv::sqrt(stddev, stddev);

	mean = mean({ 0, mean.rows - 1 }, { 0, mean.cols - 1 });
	stddev = stddev({ 0, stddev.rows - 1 }, { 0, stddev.cols - 1 });
}

cv::Mat calc_countours(const cv::Mat& src, int range)
{
	cv::Mat src_mean;
	cv::Mat src_stddev;
	cv::Mat src_median;

	mean_stddev(src, src_mean, src_stddev, range);
	cv::medianBlur(src, src_median, range);
	src_median.convertTo(src_median, CV_64FC3);

	cv::Mat countours;
	cv::divide(src_median.mul(src_stddev), src_mean, countours);
	countours.convertTo(countours, CV_8UC3);

	return countours;
}

void calc_local_max_deviation( float* data, uint32_t size, float dev_border, float& max, float& deviation )
{
    uint32_t max_loc = 0;
    max = data[0];

    for(uint32_t i = 1; i < size; i++)
    {
        if (max < data[i])
        {
            max = data[i];
            max_loc = i;
        }
    }

    uint32_t border_value = (float)max * dev_border;

    uint32_t left_loc = max_loc;
    uint32_t left = max;
    while ( left_loc > 0 && left > border_value )
    {
        left_loc--;
        left = data[left_loc];
    }

    uint32_t right_loc = max_loc;
    uint32_t right = max;
    while ( right_loc < size && right > border_value )
    {
        right_loc++;
        right = data[right_loc];
    }

    deviation = std::max((max_loc - left_loc), (right_loc - max_loc));
}

cv::Mat generate_by_row( const std::vector<float>& row )
{
    cv::Mat res( row.size(), row.size(), CV_32F );
    for( uint i = 0; i < row.size(); i++ )
    {
        for( uint j = 0; j < row.size(); j++ )
        {
            res.at<float>( j, i ) = row[i];
        }
    }
    return res;
}

bool read_frame(cv::Mat& src, cv::VideoCapture& capture)
{
    capture >> src;

    if (src.empty())
        return false;

    cv::cvtColor(src, src, CV_BGR2GRAY);

    return true;
}

cv::Mat calc_crosses(const cv::Mat& src)
{
    static auto vertical_filter = generate_by_row( {0.0f, 3.0f, -6.0f, 3.0f, 0.0f}
                /*{ 0.0f, 1.0f, 4.0f, -1.0f, -8.0f, -1.0f, 4.0f, 1.0f, 0.0f }*/);
    static auto horizontal_filter = vertical_filter.t();
    cv::Mat crosses;
    cv::Mat h_lines, v_lines;

    cv::filter2D(src, h_lines, -1, horizontal_filter);
    cv::filter2D(src, v_lines, -1, vertical_filter);
    cv::threshold(v_lines, v_lines, 200, 255, cv::THRESH_TOZERO);
    cv::threshold(h_lines, h_lines, 200, 255, cv::THRESH_TOZERO);
    cv::bitwise_and(h_lines, v_lines, crosses);

    return crosses;
}
