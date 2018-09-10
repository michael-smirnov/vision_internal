#pragma once

#include "types.h"

namespace vision
{    
    class SingularPointsFinder
    {
    private:
        Mat _image;
        const int _max_angles_difference_in_cluster = 10;
        
    public:
        SingularPointsFinder( const Mat& image );

        Mat calc_countour_angles() const;
        Mat calc_singular_map() const;

    private:
        float cluster_weight( int optimal_values_count, int values_count, int cluster_size ) const;
    }; 
}