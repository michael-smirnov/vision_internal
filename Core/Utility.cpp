#include "Utility.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace vision
{
    void Utility::add_histogram( const Mat& histogram,
                                 const Scalar& color,
                                 Mat& image )
    {
        int hist_w = 256;
        int hist_h = image.rows - image.rows * 0.01;
        int bin_w = cvRound((double)hist_w / 256);

        cv::normalize( histogram, histogram, 0, image.rows, cv::NORM_MINMAX, -1, Mat() );

        for( int i = 1; i < 256; i++ )
            cv::line( image,
                      Point( bin_w*(i - 1), hist_h - cvRound(histogram.at<float>(i - 1)) ),
                      Point( bin_w*(i), hist_h - cvRound(histogram.at<float>(i)) ),
                      color,
                      2,
                      cv::LINE_8,
                      0 );
    }
}