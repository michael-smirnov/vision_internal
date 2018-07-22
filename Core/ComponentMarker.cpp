#include "ComponentMarker.h"

#include <cstdint>

namespace vision
{
    ComponentMarker::ComponentMarker(const cv::Mat& image)
    {
        cv::Mat labels;

        cv::threshold(image, labels, 20, 255, cv::THRESH_TOZERO);
        labels.convertTo(labels, CV_16S);
        labels *= -1;

        uint8_t current_label = 0;
        for(int row = 0; row < labels.rows; row++)
        {
            for(int col = 0; col < labels.cols; col++)
            {
                if( labels.at<int16_t>(row, col) < 0 )
                {
                    current_label += 1;
                    _areas.push_back({col, row, 0, 0});
                    search(labels, current_label, {col, row});
                }
            }
        }
    }

    std::vector<cv::Rect2i> ComponentMarker::get_area_boxes()
    {
        return _areas;
    }

    void ComponentMarker::search(cv::Mat& labels, uint8_t current_label, cv::Point2i origin)
    {
        auto rect = _areas[ current_label-1 ];
        cv::Rect2i tmp;
        cv::Point2i rd = { std::max(rect.x+rect.width, origin.x), std::max(rect.y+rect.height, origin.y) };

        tmp.x = std::min(rect.x, origin.x);
        tmp.y = std::min(rect.y, origin.y);
        tmp.width = rd.x - tmp.x;
        tmp.height = rd.y - tmp.y;

        _areas[ current_label-1 ] = tmp;

        labels.at<int16_t>(origin) = current_label;
        auto ns = neighbors(origin, labels.rows, labels.cols);
        for( auto p : ns )
        {
            if( labels.at<int16_t>(p) < 0 )
                search(labels, current_label, p);
        }
    }

    std::vector<cv::Point2i> ComponentMarker::neighbors(cv::Point2i origin, int rows, int cols)
    {
        std::vector<cv::Point2i> res = {};

        if( origin.y > 0 )
        {
            if( origin.x > 0 )
                res.push_back( {origin.x-1, origin.y-1} );
            res.push_back( {origin.x, origin.y-1} );
            if( origin.x < cols - 1 )
                res.push_back( {origin.x+1, origin.y-1} );
        }

        if( origin.x > 0 )
            res.push_back( {origin.x-1, origin.y} );
        if( origin.x < cols - 1 )
            res.push_back( {origin.x+1, origin.y} );

        if( origin.y < rows - 1 )
        {
            if( origin.x > 0 )
                res.push_back( {origin.x-1, origin.y+1} );
            res.push_back( {origin.x, origin.y+1} );
            if( origin.x < cols - 1 )
                res.push_back( {origin.x+1, origin.y+1} );
        }

        return res;
    }
}
