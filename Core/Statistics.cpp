#include "Statistics.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace cv;

namespace vision
{
    float Statistics::median( const Mat& m )
    {
        int channels[] = { 0 };
        int histSize[] = { 256 };
        float ranges_array[] = { 0.0f, 256.0f };
        const float* ranges = { ranges_array };

        Mat histogram;
        calcHist( &m, 1, channels, Mat(), histogram, 1, histSize, &ranges );

        float median = -1.0f;
        int in_bins = 0;
        float middle = (m.rows * m.cols) / 2.0f;
        for( int i = 0; i < *histSize; i++ )
        {
            in_bins += cvRound( histogram.at<float>(i) );
            if( in_bins >= middle )
            {
                median = i;
                break;
            }
        }

        return median;
    }

    void Statistics::local_max_and_stddev( const float* data, 
                                           uint32_t size, 
                                           float stddev_value_threshold, 
                                           float& max, 
                                           float& deviation )
    {
        uint32_t max_loc = 0;
        max = data[0];

        for(uint32_t i = 1; i < size; i++)
        {
            if (max < data[i])
            {
                max = data[i];
                max_loc = i;
            }
        }

        float threshold_value = max * stddev_value_threshold;

        uint32_t left_loc = max_loc;
        float left = max;
        while ( left_loc > 0 && left > threshold_value )
        {
            left_loc--;
            left = data[left_loc];
        }

        uint32_t right_loc = max_loc;
        float right = max;
        while ( right_loc < size && right > threshold_value )
        {
            right_loc++;
            right = data[right_loc];
        }

        deviation = std::max((max_loc - left_loc), (right_loc - max_loc));
    }
}