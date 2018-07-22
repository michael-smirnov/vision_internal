#pragma once

#include <opencv2/imgproc.hpp>
#include <vector>

namespace vision
{
    class ComponentMarker
    {
    public:
        ComponentMarker(const cv::Mat& image);
        std::vector<cv::Rect2i> get_area_boxes();

    private:
        std::vector<cv::Rect2i> _areas;

    private:
        void search(cv::Mat& labels, uint8_t current_label, cv::Point2i origin);
        std::vector<cv::Point2i> neighbors(cv::Point2i origin, int rows, int cols);
    };
}
