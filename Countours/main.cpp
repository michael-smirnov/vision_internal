#include "ComponentMarker.h"
#include "TraceFinder.h"
#include "ImageProcessor.h"
#include "MathItemGenerator.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <exception>

using namespace std;
using namespace cv;
using namespace vision;

const string wnd_countours = "countours";
const string wnd_area = "area";
const string wnd_trace = "trace";
const string wnd_crosses = "crosses";

std::vector<int> get_angle_histogram( const cv::Mat& m, uint32_t histogram_length = 36 )
{
    vector<int> v( histogram_length, 0 );

    for( int i = 0; i < m.rows; i++ )
    {
        for( int j = 0; j < m.cols; j++ )
        {
            auto value = m.at<int16_t>( i, j );
            if( value >= 0 )
            {
                float normalized_value = static_cast<float>(value) / 360.0f;
                int bin = normalized_value * (histogram_length-1);
                v[ bin ]++;
            }
        }
    }

    return v;
}

int get_median_index_only_by_increment( std::vector<int> histogram, int half_window, int start_index )
{
    int cur_index = start_index;

    int sum = histogram[cur_index];
    for( int j = cur_index + 1; j < histogram.size() && j < cur_index + 1 + half_window; j++ )
        sum += histogram[j];

    int k = histogram[cur_index];
    while( k < sum / 2 )
    {
        cur_index++;
        k += histogram[cur_index];
    }

    return cur_index;
}

int get_median_index_by_center( std::vector<int> histogram, int half_window, int center_index )
{
    int last_histogram_index = histogram.size() - 1;

    int start_index = max( 0, center_index - half_window );
    int end_index = min( last_histogram_index, center_index + half_window );

    int sum = 0;
    for( int i = start_index; i <= end_index; i++ )
        sum += histogram[i];

    if( sum == 1 )
        return center_index;

    int cur_index = start_index;

    int k = histogram[cur_index];
    while( k < sum / 2 )
    {
        cur_index++;
        k += histogram[cur_index];
    }

    return cur_index;
}

struct angle_histogram_moda
{
    int angle_histogram_index;
    float values_percent_in_moda;
};

std::vector<angle_histogram_moda> get_most_valuable_arguments( std::vector<int> histogram, int half_window )
{
    std::vector<angle_histogram_moda> args;

    int sum_histogram = 0;
    for( int j = 0; j < histogram.size(); j++ )
        sum_histogram += histogram[j];

    int i = 0;
    while( i < histogram.size() )
    {
        if( histogram[i] == 0 )
            i++;
        else
        {
            i = get_median_index_only_by_increment( histogram, half_window, i );
            i = get_median_index_by_center( histogram, half_window, i );

            int moda_sum = 0;
            for( int k = std::max( 0, i - half_window ); 
                 k < std::min( (int)histogram.size(), i + half_window + 1 ); 
                 k++ )
            {
                moda_sum += histogram[k];
            }

            args.push_back( { i, (float)moda_sum / sum_histogram } );
            i += half_window + 1;
        }
    }

    return args;
}

void mouse_callback( int event, int x, int y, int flag, void* param )
{
    if (event == EVENT_MOUSEMOVE)
    {
        cout << "(" << x << ", " << y << ")" << endl;
    }
}

int main(int argc, char* argv[])
{
    namedWindow(wnd_countours, WINDOW_NORMAL);
    namedWindow(wnd_area, WINDOW_NORMAL);
    namedWindow(wnd_trace, WINDOW_NORMAL);
    namedWindow(wnd_crosses, WINDOW_NORMAL);

    Mat img = imread(argv[1]);
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

    Mat countours = ImageProcessor::countours( img_gray, 3 );
    Mat tr = Mat::zeros( countours.rows, countours.cols, CV_8U );

    auto horizontal_filter = MathItemGenerator::matrix_by_row( { 0.0f, 1.0f, 4.0f, -1.0f, -8.0f, -1.0f, 4.0f, 1.0f, 0.0f } );
	auto vertical_filter = horizontal_filter.t();

    Mat crosses = ImageProcessor::crosses( img_gray, horizontal_filter, vertical_filter );

    Mat dx_sobel, dy_sobel, mag_sobel;
    ImageProcessor::dxdy_sobel( img_gray, 1, 1, dx_sobel, dy_sobel );

    auto angles = ImageProcessor::dxdy_argument( dx_sobel, dy_sobel );
    auto colour_angles = MathItemGenerator::color_map_matrix( angles, 0, 360 );

    Mat singulars = Mat::zeros( angles.rows, angles.cols, CV_8UC3 );

    int kernel_size = 3;
    int offset = kernel_size / 2;

    for( int i = offset; i < angles.rows - offset; i++ )
    {
        for( int j = offset; j < angles.cols - offset; j++ )
        {
            auto histogram = get_angle_histogram( angles( {j-offset, i-offset, kernel_size, kernel_size} ), 72 );
            auto peaks = get_most_valuable_arguments( histogram, 3 );

            if( peaks.size() < 2 )
                continue;

            float k = 0.0f;
            float p = 1.0f / peaks.size();
            float err = 0.0f;
            for( int m = 0; m < peaks.size(); m++ )
                err += std::pow( peaks[m].values_percent_in_moda - p, 2.0f );
            k = std::sqrt( err );

            int grade = peaks.size() > 5 ? 3 : peaks.size() - 2;
            uchar peaks_grades[] = { 20, 50, 150, 255 };

            uchar color = peaks_grades[grade];
            singulars.at<Vec3b>( i, j ) = { (uchar)(color - k * color), 
                                            (uchar)(color - k * color), 
                                            (uchar)(color - k * color) };
        }
    }

    setMouseCallback( wnd_countours, mouse_callback );

    imshow( wnd_countours, countours );
    imshow( wnd_trace, colour_angles );
    imshow( wnd_crosses, singulars );

    waitKey();
    return 0;
}
