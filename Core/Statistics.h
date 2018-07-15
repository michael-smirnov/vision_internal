#pragma once

#include "types.h"

namespace vision
{
    class Statistics
    {
    public:
        static float median( const Mat& m );
        static void local_max_and_stddev( const float* data, 
                                          uint32_t size, 
                                          float stddev_value_threshold, 
                                          float& max, 
                                          float& deviation );
    private:
        Statistics() {}
    };
}