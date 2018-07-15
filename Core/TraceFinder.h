#pragma once

#include "types.h"
#include "trace.h"
#include <unordered_set>
#include <optional>
#include <vector>

namespace vision
{
    class TraceFinder
    {
    public:
        TraceFinder( const Mat& countours,
                     const Mat& singulars );

        std::vector<trace> find_traces();

        std::optional<Point> next_singular_point();
        Point max_gradient_singular_point_in_area( const Point& start_singular_point );
        std::vector<int> local_directions( const Point& origin );
        int global_direction( const Point& origin, const Vec2f direction );
        void traverse( trace& );
        Point get_last_point( trace& );
        void mark_singular_area( const Point& origin );

    private:
        void max_gradient_direction( const Matx33d& m, double (&directions)[8], int& max_index );
        Vec2f closest_direction( const Vec2f& v, int degree );
        std::vector<Vec2f> all_directions( int degree );

    private:
        class PointHash
        {
        public:
            size_t operator()( const cv::Point& p ) const;
        };

    private:
        const int _singular_point_threshold;
        std::vector<Vec2f> _area3x3_vectors;

        Mat _countours;
        Mat _singulars;

        Mat _tr;
        Mat _ar;

        std::vector<trace> _traces;

        std::unordered_set<Point, PointHash> _spent_singular_points;
        std::unordered_set<Point, PointHash> _spent_regular_points;
    };
}
