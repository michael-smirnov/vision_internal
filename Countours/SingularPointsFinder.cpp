#include "SingularPointsFinder.h"
#include "ImageProcessor.h"
#include "HistogramModaCalculator.h"
#include "Statistics.h"
#include "MathItemGenerator.h"

#include <vector>
#include <opencv2/highgui.hpp>

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

        Mat singulars = Mat::zeros( angles.rows, angles.cols, CV_8U );

        int kernel_size = 5;
        int offset = kernel_size / 2;

        for( int i = offset; i < angles.rows - offset; i++ )
        {
            for( int j = offset; j < angles.cols - offset; j++ )
            {
                auto area = angles( {j-offset, i-offset, kernel_size, kernel_size} );

                /*if( i == 362 && j == 295 )
                {
                     auto colour_angles = MathItemGenerator::color_map_matrix( area, 0, 360 );
                     cv::imshow("trace", colour_angles);
                     cv::waitKey();
                }*/

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
                        int distance_right = sorted_area[i+1] - sorted_area[i];

                        if( /*distance_left <= distance_right &&*/ distance_left < _max_angles_difference_in_cluster )
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
                    int v1 = sorted_area.size() - clusters[i] * clusters.size();
                    int v2 = sorted_area.size();

                    float cluster_weight = static_cast<float>(v1*v1) / static_cast<float>(v2*v2);

                    singular_point_weight += cluster_weight;
                }
                singular_point_weight *= (clusters.size() - 1);

                singulars.at<uint8_t>( i, j ) = static_cast<uint8_t>( singular_point_weight*10 );
            }
        }

        return singulars;
    }
}
