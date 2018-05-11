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
	double n = window_size * window_size;

	cv::Mat kernel = cv::Mat::zeros(window_size + 1, window_size + 1, CV_64F);
	kernel.at<double>(0, 0) = 1.0 / n;
	kernel.at<double>(0, window_size) = -1.0 / n;
	kernel.at<double>(window_size, 0) = -1.0 / n;
	kernel.at<double>(window_size, window_size) = 1.0 / n;

	cv::Mat sum;
	cv::Mat sqsum;

	cv::integral(src, sum, sqsum);
	sum.convertTo(sum, CV_64FC3);

	cv::filter2D(sum, mean, -1, kernel);
	cv::filter2D(sqsum, stddev, -1, kernel);

	mean.convertTo(mean, sqsum.type());
	stddev = stddev - mean.mul(mean);
	cv::sqrt(stddev, stddev);

	auto offset = window_size / 2 + 1;
	auto m = mean({ offset, mean.rows - offset - 1 }, { offset, mean.cols - offset - 1 }).clone();
	auto s = stddev({ offset, stddev.rows - offset - 1 }, { offset, stddev.cols - offset - 1 }).clone();

	cv::copyMakeBorder(s, stddev, offset, offset, offset, offset, cv::BORDER_CONSTANT, { 0.0 });
	cv::copyMakeBorder(m, mean, offset, offset, offset, offset, cv::BORDER_CONSTANT, { 0.0 });
}

cv::Mat calc_countours(const cv::Mat& src, int range)
{
	cv::Mat src_mean;
	cv::Mat src_stddev;
	cv::Mat src_median;

	mean_stddev(src, src_mean, src_stddev, range);
	cv::medianBlur(src, src_median, range);
	src_median.convertTo(src_median, CV_64FC3);

	cv::Mat countours = src_stddev;
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

    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

    return true;
}

cv::Mat calc_crosses(const cv::Mat& src)
{
    static auto vertical_filter = generate_by_row( 
                /*{0.0f, 3.0f, -6.0f, 3.0f, 0.0f}*/
                { 0.0f, 1.0f, 4.0f, -1.0f, -8.0f, -1.0f, 4.0f, 1.0f, 0.0f }
	);
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

cv::Mat derivative(const cv::Mat& src, int window_size)
{
	std::vector<float> v;

	for (int i = 0; i < window_size / 2; i++)
		v.push_back(-1.0f);
	v.push_back(0.0f);
	for (int i = 0; i < window_size / 2; i++)
		v.push_back(1.0f);

	cv::Mat vert = generate_by_row(v);
	cv::Mat hori = vert.t();

	cv::Mat conv_src;
	src.convertTo(conv_src, CV_32F);

	cv::Mat dx, dy;
	cv::filter2D(conv_src, dx, -1, vert);
	cv::filter2D(conv_src, dy, -1, hori);
	dx /= (window_size - 1);
	dy /= (window_size - 1);

	cv::Mat magn = dx.mul(dx) + dy.mul(dy);
	cv::Mat res;
	cv::sqrt(magn, res);

	return res;
}

void max_gradient_direction(const cv::Matx33d& m, double (&directions)[8], int& max_index)
{
    double dirs[] = { m(0,0), m(0,1), m(0,2), m(1,0), m(1,2), m(2,0), m(2,1), m(2,2) };
    double max = dirs[0] / m(1,1);
    for(int i = 0; i < 8; i++)
    {
        directions[i] = dirs[i] / m(1,1);
        if( directions[i] > max )
        {
            max = directions[i];
            max_index = i;
        }
    }
}

cv::Vec2f scale_min_component_to_one(const cv::Vec2f v)
{
    float min = std::min(std::abs(v[0]), std::abs(v[1]));
    return { v[0] / min, v[1] / min};
}

cv::Vec2f closest_direction(const cv::Vec2f v, int degree)
{
    int offset = degree / 2;
    float max_value = std::max(std::abs(v[0]), std::abs(v[1]));

    return cv::Vec2f{ v[0]/max_value*offset, v[1]/max_value*offset };
}

std::vector<cv::Vec2f> all_directions( int degree )
{
    std::vector<cv::Vec2f> directions = {};
    int offset = degree / 2;

    for(int col = -offset; col <= offset; col++)
            directions.push_back({col, -offset});

    for(int row = -offset+1; row <= offset-1; row++)
    {
            directions.push_back({-offset, row});
            directions.push_back({offset, row});
    }
    for(int col = -offset; col <= offset; col++)
            directions.push_back({col, offset});

    return directions;
}
