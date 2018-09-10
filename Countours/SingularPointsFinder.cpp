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

        int kernel_size = 5;
        int offset = kernel_size / 2;

        for( int i = offset; i < angles.rows - offset; i++ )
        {
            for( int j = offset; j < angles.cols - offset; j++ )
            {
                auto area = angles( {j-offset, i-offset, kernel_size, kernel_size} );

                std::vector<int> sorted_area = {};
                //counting sort
                {
                    const int max_angle = 360;

                    auto histogram = ImageProcessor::get_histogram<int16_t>( area, 0, max_angle, max_angle + 1 ); 
                    for( int i = 0; i < max_angle + 1; i++ )
                    {
                        for( int j = 0; j < histogram[i]; j++ )
                            sorted_area.push_back(i);
                    }
                }
                
                if( sorted_area.size() < 2 )
                    continue;

                std::vector<int> clusters = { 1 };
                //merging values into clusters
                {
                    for( int i = 1; i < sorted_area.size()-1; i++ )
                    {
                        int distance_left = sorted_area[i] - sorted_area[i-1];

                        if( distance_left < _max_angles_difference_in_cluster )
                            clusters[ clusters.size()-1 ]++;
                        else
                            clusters.push_back( 1 );
                    }

                    int last = sorted_area[ sorted_area.size()-1 ];
                    int before_last = sorted_area[ sorted_area.size()-2 ]; 

                    if( last - before_last < _max_angles_difference_in_cluster )
                        clusters[ clusters.size()-1 ]++;
                    else
                        clusters.push_back( 1 );
                }

                float singular_point_weight = 0.0f;
                for( int i = 0; i < clusters.size(); i++ )
                {
                    float cluster_weight = this->cluster_weight( sorted_area.size() / clusters.size(),
                                                                 clusters[i],
                                                                 clusters.size() );

                    singular_point_weight += cluster_weight;
                }
                singular_point_weight *= sorted_area.size();

                singulars.at<float>( i, j ) = singular_point_weight;
            }
        }

        return singulars;
    }

    float SingularPointsFinder::cluster_weight( int optimal_values_count, int values_count, int cluster_size ) const
    {
        int difference = std::abs(optimal_values_count - values_count);
        float percent_diff = static_cast<float>(difference) / cluster_size;

        float err = 0.0f;
        if( percent_diff < 0.1f )
            err = 0.0f;
        else if( percent_diff < 0.15f )
            err = 0.1f;
        else if( percent_diff < 0.2f )
            err = 0.3f;
        else if( percent_diff < 0.3f )
            err = 0.5f;
        else if( percent_diff < 0.5f )
            err = 0.7f;
        else if( percent_diff < 0.6f )
            err = 0.9f;
        else 
            err = 1.0f;

        return 1.0f - err;
    }
}
