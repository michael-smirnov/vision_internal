#include "SingularPointsFinder.h"
#include "ImageProcessor.h"
#include "HistogramModaCalculator.h"
#include "Statistics.h"
#include "MathItemGenerator.h"

#include <vector>
#include <opencv2/highgui.hpp>
#include <iostream>

namespace vision
{
    SingularPointsFinder::SingularPointsFinder( const Mat& image )
    : _image( image.clone() )
    {}

    Mat SingularPointsFinder::calc_countour_angles() const
    {
        Mat dx_sobel, dy_sobel;
        
        ImageProcessor::dxdy_sobel( _image, 1, 1, dx_sobel, dy_sobel );
        return ImageProcessor::dxdy_argument( dx_sobel, dy_sobel );
    }

    Mat SingularPointsFinder::calc_singular_map() const
    {
        auto angles = calc_countour_angles();

        Mat singulars = Mat::zeros( angles.rows, angles.cols, CV_32F );

        int kernel_size = 3;
        int offset = kernel_size / 2;

        for( int i = offset; i < angles.rows - offset; i++ )
        {
            for( int j = offset; j < angles.cols - offset; j++ )
            {
                auto area = angles( {j-offset, i-offset, kernel_size, kernel_size} );

                if( angles.at<int16_t>(i, j) < 0 )
                    continue;

                const int max_angle = 360;

                std::vector<int> clusters = ImageProcessor::histogram<int16_t>( area, 0, max_angle, 8 );
                
                const float min_elements_in_area = kernel_size * kernel_size * 0.5f;
                int elements_in_area = Statistics::sum( clusters );
                
                if( elements_in_area < min_elements_in_area )
                    continue;

                int max_element = 0;
                for( int element : clusters )
                    max_element = std::max( max_element, element );

                float singular_point_weight = 0.0f;
                for( int i = 0; i < clusters.size(); i++ )
                {
                    float cluster_weight = static_cast<float>( clusters[i] ) / max_element;

                    singular_point_weight += cluster_weight;
                }

                if( singular_point_weight < 1.5f )
                    singular_point_weight = 0.0f;

                float k = static_cast<float>( elements_in_area ) / (kernel_size * kernel_size);
                singulars.at<float>( i, j ) = singular_point_weight * k;
            }
        }

        return singulars;
    }
}
