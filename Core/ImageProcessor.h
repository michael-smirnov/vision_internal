#pragma once

#include "types.h"

#include <vector>

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

    template< typename VALUE_TYPE >
    static std::vector<int> histogram( const Mat& m, 
                                           const VALUE_TYPE min_value,
                                           const VALUE_TYPE max_value,
                                           uint32_t histogram_length );

    static Mat dxdy_argument( const Mat& src_dx, 
                              const Mat& src_dy, 
                              float min_magnitude = 10.0f );

    static std::vector<Point2i> local_maximums( const Mat& m, float diff_threshold = 0.0f );

    static void min_gradient_direction( Mat& min_grad_dirs, Mat& min_grad_weight, const Mat& dxdy_magn, int window_size );
    
    static Mat diff_vec_norm_by_direction( const Mat& dirs, int window_size );

  private:
    ImageProcessor() {}
  };

  template< typename VALUE_TYPE >
  std::vector<int> ImageProcessor::histogram( const cv::Mat& m, 
                                                  const VALUE_TYPE min_value,
                                                  const VALUE_TYPE max_value,
                                                  uint32_t histogram_length )
  {
      std::vector<int> v( histogram_length, 0 );

      for( int i = 0; i < m.rows; i++ )
      {
          for( int j = 0; j < m.cols; j++ )
          {
              auto value = m.at<VALUE_TYPE>( i, j );
              if( value >= min_value )
              {
                  float normalized_value = static_cast<float>(value - min_value) / max_value;
                  int bin = normalized_value * (histogram_length-1);
                  v[ bin ]++;
              }
          }
      }

      return v;
  }
}