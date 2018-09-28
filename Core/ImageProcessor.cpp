#include "ImageProcessor.h"

#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <exception>

#include <iostream>

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
        src_median.convertTo( src_median, CV_64F );

        Mat result = src_stddev;
        cv::divide( src_median.mul(src_stddev), src_mean, result, 1.0, CV_8U );

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

    std::vector<Point2i> ImageProcessor::local_maximums( const Mat& m, float diff_threshold )
    {
        std::vector<Point2i> maximums = {};

        if( m.type() != CV_32F )
            throw std::runtime_error( "invalid matrix type" );

        const int window_size = 3;
        const int offset = window_size / 2;

        for( int row = offset; row < m.rows - offset; row++ )
        {
            for( int col = offset; col < m.cols - offset; col++ )
            {
                float center_element = m.at<float>( row, col );
                bool is_maximum = true;

                for( int window_row = row - offset; window_row <= row + offset && is_maximum; window_row++ )
                {
                    for( int window_col = col - offset; window_col <= col + offset && is_maximum; window_col++ )
                    {
                        float diff = center_element - m.at<float>(window_row, window_col);
                        if( diff < diff_threshold )
                            is_maximum = false;
                    }
                }

                if( is_maximum )
                    maximums.push_back( {col, row} );
            }
        }

        return maximums;
    }

    void ImageProcessor::min_gradient_direction( Mat& min_grad_dirs, Mat& min_grad_weight, const Mat& dxdy_magn, int window_size )
    {
        if( dxdy_magn.type() != CV_64F )
            throw std::runtime_error( "invalid matrix type" );

        min_grad_dirs = Mat::zeros( dxdy_magn.rows, dxdy_magn.cols, CV_64FC2 );
        min_grad_weight = Mat::zeros( dxdy_magn.rows, dxdy_magn.cols, CV_64F );

        const int offset = window_size / 2;

        auto abs_diff_by_direction = [offset]( const Mat& area, const Vec2d& v )
        {
            auto point = Point2d{ offset, offset };
            double diff_abs = 0.0;
            double t = 1.0 / offset;
            
            for( int iteration = 1; iteration <= offset; iteration++ )
            {
                t *= iteration;
                auto next_point = Point2d{ point.x + v[0]*t, 
                                           point.y + v[1]*t };
                diff_abs += std::abs( area.at<double>(point) - area.at<double>(next_point) );
                //point = next_point;
            }
            diff_abs /= offset;

            return diff_abs;
        };

        for( int row = offset; row < dxdy_magn.rows - offset; row++ )
        {
            for( int col = offset; col < dxdy_magn.cols - offset; col++ )
            {
                std::multimap< double, Vec2d > directions;
                auto area = dxdy_magn( { col-offset, row-offset, window_size, window_size } );
                
                /*if( row >= 27 )*/
                /*{
                    std::cout << row << ";" << col << std::endl;
                    cv::imshow( "window 3", area );
                    cv::waitKey();
                }*/

                for( int area_col = 0; area_col < window_size; area_col++ )
                {
                    auto v = Vec2d{ double(area_col-offset), double(-offset) };
                    double d1 = abs_diff_by_direction( area, v );
                    double d2 = abs_diff_by_direction( area, -v );
                    double diff = ( d1 + d2 ) / 2.0;
                    directions.insert( std::make_pair( diff, v ) );
                }

                for( int area_row = 1; area_row < window_size-1; area_row++ )
                {
                    auto v = Vec2d{ double(window_size-1-offset), double(area_row-offset) };
                    double d1 = abs_diff_by_direction( area, v );
                    double d2 = abs_diff_by_direction( area, -v );
                    double diff = ( d1 + d2 ) / 2.0;
                    directions.insert( std::make_pair( diff, v ) );
                }

                int count = 0;
                for( auto i = directions.begin(); i != directions.lower_bound( 0.05 ); i++ )
                    count++;

                if( count > 0 && count < 3 )
                {
                    min_grad_dirs.at<Vec2d>( row, col ) = directions.begin()->second;
                    min_grad_weight.at<double>( row, col ) = directions.begin()->first;
                }
                else
                {
                    min_grad_dirs.at<Vec2d>( row, col ) = { 0, 0 };
                    min_grad_weight.at<double>( row, col ) = 1.0;
                }
            }
        }
    }

    Mat ImageProcessor::diff_vec_norm_by_direction( const Mat& dirs, int window_size )
    {
        Mat result = Mat::zeros( dirs.rows, dirs.cols, CV_64F );

        const int offset = window_size / 2;

        for( int row = 2*offset; row < dirs.rows - 2*offset; row++ )
        {
            for( int col = 2*offset; col < dirs.cols - 2*offset; col++ )
            {
                auto area = dirs( { col - offset, row - offset, window_size, window_size } );
                auto v = dirs.at<Vec2d>( row, col );
                double diff_vec_length = 0.0f;

                auto point = Point2d{ offset - v[0], offset - v[1] };
                Vec2d diff;
                double t = 1.0 / offset;
                
                for( int iteration = 1; iteration <= 2*offset; iteration++ )
                {
                    auto next_point = Point2d{ point.x + v[0]*t, 
                                               point.y + v[1]*t };
                    diff += area.at<Vec2d>(point) - area.at<Vec2d>(next_point);
                    point = next_point;
                }
                diff /= 2*offset;
                diff_vec_length = norm( diff );

                result.at<double>( row, col ) = diff_vec_length;
            }
        }

        return result;
    }
}