#pragma once

#include "types.h"

namespace vision
{
  class ImageProcessor
  {
  public:
    static void mean_stddev( const Mat& src, 
                             Mat& mean, 
                             Mat& stddev, 
                             int window_size );
    
    static void dxdy_sobel( const Mat& src, 
                            int dx_degree,
                            int dy_degree,
                            Mat& dx, 
                            Mat& dy );

    static Mat countours( const Mat& src, int window_size );
    
    static Mat crosses( const Mat& src, 
                        const Mat& horizontal_filter,
                        const Mat& vertical_filter,
                        int zero_value_threshold = 200 );

    static Mat dxdy_argument( const Mat& src_dx, 
                              const Mat& src_dy, 
                              float min_magnitude = 10.0f );
  private:
    ImageProcessor() {}
  };
}