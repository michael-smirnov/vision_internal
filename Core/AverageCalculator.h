#pragma once

#include "types.h"

#include <queue>
#include <opencv2/imgproc.hpp>

namespace vision
{
	class AverageCalculator
	{
	public:
		AverageCalculator(int capacity);
		~AverageCalculator();

		void add_frame(const Mat&);
		Mat get_average();

	private:
		std::queue<cv::Mat> _frames;
		Mat _average;
		const int _average_capacity;
		int _current_frames_count;
	};
}

