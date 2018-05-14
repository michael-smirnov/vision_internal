#pragma once

#include "trace.h"
#include <unordered_set>
#include <optional>

class TraceFinder
{
public:
    TraceFinder( const cv::Mat& countours,
                 const cv::Mat& singulars );

    std::vector<vision::trace> find_traces();

    std::optional<cv::Point> next_singular_point();
    cv::Point max_gradient_singular_point_in_area( const cv::Point& start_singular_point );
    std::vector<int> local_directions( const cv::Point& origin );
    int global_direction( const cv::Point& origin, const cv::Vec2f direction );
    void traverse( vision::trace& );
    cv::Point get_last_point( vision::trace& );
    void mark_singular_area( const cv::Point& origin );

private:
    class PointHash
    {
    public:
        size_t operator()( const cv::Point& p ) const;
    };

private:
    const int _singular_point_threshold;
    std::vector<cv::Vec2f> _area3x3_vectors;

    cv::Mat _countours;
    cv::Mat _singulars;

    cv::Mat _tr;
    cv::Mat _ar;

    std::vector<vision::trace> _traces;

    std::unordered_set<cv::Point, PointHash> _spent_singular_points;
    std::unordered_set<cv::Point, PointHash> _spent_regular_points;
};
