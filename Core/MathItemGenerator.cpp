#include "MathItemGenerator.h"

namespace vision
{
    Mat MathItemGenerator::matrix_by_row( const std::vector<float>& row )
    {
        Mat res( row.size(), row.size(), CV_32F );

        for( uint i = 0; i < row.size(); i++ )
            for( uint j = 0; j < row.size(); j++ )
                res.at<float>( j, i ) = row[i];

        return res;
    }

    Mat MathItemGenerator::color_map_matrix( const Mat& m, 
                                             int16_t min_value, 
                                             int16_t max_value )
    {
        Mat colored = Mat::zeros( m.rows, m.cols, CV_8UC3 );
        int16_t half_value = (max_value - min_value) / 2;

        for(int row = 0; row < m.rows; row++)
        {
            for(int col = 0; col < m.cols; col++)
            {
                int16_t value = m.at<int16_t>(row, col);
                if( value < min_value )
                    continue;

                if( value >= min_value && value < half_value )
                {
                    uint8_t r = (value) / (float)half_value * 255;
                    uint8_t g = (half_value - value) / (float)half_value * 255;
                    colored.at<Vec3b>(row, col) = { 0, g, r };
                }
                else
                {
                    uint8_t r = (max_value- value) / (float)half_value * 255;
                    uint8_t b = (value - half_value) / (float)half_value * 255;
                    colored.at<Vec3b>(row, col) = { b, 0, r };
                }
            }
        }

        return colored;
    }
}