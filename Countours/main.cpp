#include "ComponentMarker.h"
#include "TraceFinder.h"
#include "ImageProcessor.h"
#include "MathItemGenerator.h"
#include "Statistics.h"
#include "HistogramModaCalculator.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace vision;

const string wnd_countours = "countours";
const string wnd_area = "area";
const string wnd_trace = "trace";
const string wnd_crosses = "crosses";

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
            auto histogram = ImageProcessor::get_histogram<int16_t>( angles( {j-offset, i-offset, kernel_size, kernel_size} ), 0, 360, 72 );
            HistogramModaCalculator calculator( histogram, 3 );

            auto modas = calculator.calculate_modas();

            int histogram_sum = Statistics::sum( histogram );

            if( modas.size() < 2 )
                continue;

            float k = 0.0f;
            float p = 1.0f / modas.size();
            float err = 0.0f;
            for( int m = 0; m < modas.size(); m++ )
                err += std::pow( static_cast<float>( modas[m].values_count ) / histogram_sum  - p, 2.0f );
            k = std::sqrt( err );

            int grade = modas.size() > 5 ? 3 : modas.size() - 2;
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
