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
    }; 
}