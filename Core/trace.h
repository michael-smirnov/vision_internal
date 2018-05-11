#pragma once

#include <opencv2/core.hpp>
#include <vector>

struct trace
{
    cv::Point start;
    cv::Point end;
    std::vector<cv::Vec2i> directions;
    bool completed = false;
};
