#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace vision
{
    struct trace
    {
        cv::Point start;
        std::vector<cv::Vec2f> directions;
    };
}
