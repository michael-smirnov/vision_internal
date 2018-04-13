#include "AverageCalculator.h"

AverageCalculator::AverageCalculator(int capacity)
	: _average_capacity(capacity)
	, _current_frames_count(0)
{}

AverageCalculator::~AverageCalculator()
{}

void AverageCalculator::add_frame(const cv::Mat& frame)
{
	if (_current_frames_count == 0)
		_average = cv::Mat::zeros(frame.size(), CV_32FC3);

	cv::Mat tmp;
	frame.convertTo(tmp, CV_32FC3);

	if (_current_frames_count < _average_capacity)
	{
		cv::add(tmp, _average, _average);

		_frames.push(tmp);
		_current_frames_count++;
	}
	else
	{
		cv::subtract(_average, _frames.front(), _average);
		cv::add(tmp, _average, _average);

		_frames.pop();
		_frames.push(tmp);
	}
}

cv::Mat AverageCalculator::get_average()
{
	cv::Mat res = _average / _current_frames_count;
	res.convertTo(res, CV_8UC3);
	return res;
}
