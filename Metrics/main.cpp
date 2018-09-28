#include "types.h"
#include "ImageProcessor.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <iostream>

using namespace vision;
using namespace cv;

static const std::string window_main = "main window"; 
static const std::string window_2 = "window 2";
static const std::string window_3 = "window 3";
static const std::string window_4 = "window 4";

int main()
{
    namedWindow( window_main, WINDOW_NORMAL );
    namedWindow( window_2, WINDOW_NORMAL );
    namedWindow( window_3, WINDOW_NORMAL );
    namedWindow( window_4, WINDOW_NORMAL );

    Mat image = Mat::zeros( 200, 200, CV_8U );
    Mat noise = Mat::zeros( image.rows, image.cols, CV_8U ); 

    line( image, { 5, 5 }, { 150, 150 }, { 100, 100, 100 }, 1 );
    line( image, { 5, 150 }, { 150, 5 }, { 255, 255, 255 }, 1 );

    randn( noise, 0, 20 );
    //image += noise;

    double min;
    double max;

    auto countours = ImageProcessor::countours( image, 3 );
    Mat countours_double;
    //countours.convertTo( countours_double, CV_64F );

    Mat dx;
    Mat dy;

////////////////////////////////////////////////////////////////////
    ImageProcessor::dxdy_sobel( image, 1, 1, dx, dy );
    cv::sqrt( dx.mul(dx) + dy.mul(dy), countours_double );

    minMaxLoc( countours_double, &min, &max );
    countours_double = (countours_double - min) / (max - min);
////////////////////////////////////////////////////////////////////

    Mat magn_dx_dy;
    ImageProcessor::dxdy_sobel( image, 2, 2, dx, dy );
    cv::sqrt( dx.mul(dx) + dy.mul(dy), magn_dx_dy );

    minMaxLoc( magn_dx_dy, &min, &max );
    magn_dx_dy = (magn_dx_dy - min) / (max - min);

    Mat min_grad_dir;
    Mat min_grad_weight;
    const int window_size = 5;
    const int offset = window_size / 2;

    ImageProcessor::min_gradient_direction( min_grad_dir, min_grad_weight, magn_dx_dy, window_size );
    auto diff_norms = ImageProcessor::diff_vec_norm_by_direction( min_grad_dir, window_size );

    minMaxLoc( min_grad_weight, &min, &max );
    min_grad_weight = (min_grad_weight - min) / (max - min);

    minMaxLoc( diff_norms, &min, &max );
    diff_norms = (diff_norms - min) / (max - min);

    Mat min_grad_dirs_image = Mat::zeros( image.rows, image.cols, CV_8U );
    for( int row = 2*offset; row < min_grad_dir.rows - 2*offset; row++ )
    {
        for( int col = 2*offset; col < min_grad_dir.cols - 2*offset; col++ )
        {
            auto v = min_grad_dir.at<Vec2d>( row, col );

            double diff_vec_length = ( 1.0 - diff_norms.at<double>( row, col ) );
            double weight = ( 1.0 - min_grad_weight.at<double>( row, col ) );
            /*double img_color = magn_dx_dy.at<double>( row, col );
            
            if( img_color > 0.2 )
                img_color = 1.0;
            else
                img_color = 0.0;
            */
            double color = 255.0 * weight * diff_vec_length /** img_color*/;
            line( min_grad_dirs_image, { col, row }, { int(col+v[0]), int(row+v[1]) }, { color, color, color }, 1 );
        }
    }

    imshow( window_main, image );
    imshow( window_2, min_grad_dirs_image );
    imshow( window_3, countours_double );
    imshow( window_4, magn_dx_dy );

    waitKey();
    return 0;
}