#pragma once

#include <queue>
#include <opencv2/imgproc.hpp>


class AverageCalculator
{
public:
	AverageCalculator(int capacity);
	~AverageCalculator();

	void add_frame(const cv::Mat&);
	cv::Mat get_average();

private:
	std::queue<cv::Mat> _frames;
	cv::Mat _average;
	const int _average_capacity;
	int _current_frames_count;
};

