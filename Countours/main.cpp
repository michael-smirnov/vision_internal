#include "ImageProcessor.h"
#include "MathItemGenerator.h"
#include "SingularPointsFinder.h"

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

    SingularPointsFinder singularsFinder( img_gray );

    auto angles = singularsFinder.calc_countour_angles();
    auto countours = ImageProcessor::countours( img_gray, 3 );
    auto colour_angles = MathItemGenerator::color_map_matrix( angles, 0, 360 );
    auto singulars = singularsFinder.calc_singular_map();

    double max_singular = 0.0f;
    minMaxLoc( singulars, nullptr, &max_singular );

    Mat singular_map = singulars / max_singular;

    setMouseCallback( wnd_crosses, mouse_callback );

    imshow( wnd_countours, countours );
    imshow( wnd_trace, colour_angles );
    imshow( wnd_crosses, singular_map );

    waitKey();
    return 0;
}
