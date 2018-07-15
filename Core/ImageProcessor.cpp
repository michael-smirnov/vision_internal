#include "ImageProcessor.h"

#include <opencv2/imgproc.hpp>
#include <exception>

namespace vision
{
    void ImageProcessor::mean_stddev( const Mat& src, 
                                      Mat& mean, 
                                      Mat& stddev, 
                                      int window_size )
    {
        if( window_size < 1 )
            throw std::runtime_error( "invalid window size" );
        
        if( src.rows < window_size + 1 || src.cols < window_size + 1 )
            throw std::runtime_error( "window size is too big for image" );
        
        double n = window_size * window_size;

        Mat kernel = Mat::zeros( window_size + 1, window_size + 1, CV_64F );

        kernel.at<double>( 0, 0 ) = 1.0 / n;
        kernel.at<double>( 0, window_size ) = -1.0 / n;
        kernel.at<double>( window_size, 0 ) = -1.0 / n;
        kernel.at<double>( window_size, window_size ) = 1.0 / n;

        Mat sum;
        Mat sqsum;

        cv::integral( src, sum, sqsum, CV_64FC3, CV_64FC3 );

        cv::filter2D( sum, mean, -1, kernel );
        cv::filter2D( sqsum, stddev, -1, kernel );

        stddev = stddev - mean.mul( mean );
        cv::sqrt( stddev, stddev );

        auto offset = window_size / 2 + 1;
        auto mean_without_border = mean( { offset, mean.rows - offset - 1 }, { offset, mean.cols - offset - 1 } ).clone();
        auto stddev_without_border = stddev( { offset, stddev.rows - offset - 1 }, { offset, stddev.cols - offset - 1 } ).clone();

        cv::copyMakeBorder( stddev_without_border, stddev, offset, offset, offset, offset, cv::BORDER_CONSTANT, { 0.0 } );
        cv::copyMakeBorder( mean_without_border, mean, offset, offset, offset, offset, cv::BORDER_CONSTANT, { 0.0 } );
    }

    void ImageProcessor::dxdy_sobel( const Mat& src, 
                                     int dx_degree,
                                     int dy_degree,
                                     Mat& dx, 
                                     Mat& dy )
    {
        if( dx_degree < 1 || dy_degree < 1 )
            throw std::runtime_error( "invalid dxdy degree" );

        cv::Sobel( src, dx, CV_64F, dx_degree, 0 );
        cv::Sobel( src, dy, CV_64F, 0, dy_degree );
    }

    Mat ImageProcessor::countours( const Mat& src, int window_size )
    {
        if( window_size < 1 )
            throw std::runtime_error( "invalid window size" );

        Mat src_mean;
        Mat src_stddev;
        Mat src_median;

        ImageProcessor::mean_stddev( src, src_mean, src_stddev, window_size );
        cv::medianBlur( src, src_median, window_size );
        src_median.convertTo( src_median, CV_64FC3 );

        Mat result = src_stddev;
        cv::divide( src_median.mul(src_stddev), src_mean, result, 1.0, CV_8UC3 );

        return result;
    }

    Mat ImageProcessor::crosses( const Mat& src, 
                                 const Mat& horizontal_filter,
                                 const Mat& vertical_filter,
                                 int zero_value_threshold )
    {
        Mat crosses;
        Mat h_lines, v_lines;

        cv::filter2D( src, h_lines, -1, horizontal_filter );
        cv::filter2D( src, v_lines, -1, vertical_filter );

        cv::threshold( v_lines, v_lines, zero_value_threshold, 255, cv::THRESH_TOZERO );
        cv::threshold( h_lines, h_lines, zero_value_threshold, 255, cv::THRESH_TOZERO );
        cv::bitwise_and( h_lines, v_lines, crosses );

        return crosses;
    }

    Mat ImageProcessor::dxdy_argument( const Mat& src_dx, 
                                       const Mat& src_dy, 
                                       float min_magnitude )
    {
        if( src_dx.cols != src_dy.cols || src_dx.rows != src_dy.rows )
            throw std::runtime_error( "invalid derivative sizes" );

        Mat argument = Mat::zeros( src_dx.rows, src_dx.cols, CV_16S );
        for( int row = 0; row < argument.rows; row++ )
        {
            for( int col = 0; col < argument.cols; col++ )
            {
                double dx_value = src_dx.at<double>( row, col );
                double dy_value = src_dy.at<double>( row, col );

                Vec2d dxdy = { dx_value, dy_value };
                if( norm(dxdy) < min_magnitude )
                {
                    argument.at<int16_t>( row, col ) = -1;
                    continue;
                }

                dxdy = normalize( dxdy );

                if( dxdy[1] < 0.0 )
                    argument.at<int16_t>( row, col ) = (1 - dxdy[0]) / 2.0 * 180;
                else
                    argument.at<int16_t>( row, col ) = (1 + dxdy[0]) / 2.0 * 180 + 180;
            }
        }

        return argument;
    }
}