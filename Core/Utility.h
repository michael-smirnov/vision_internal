#pragma once

#include "types.h"

namespace vision
{
    class Utility
    {
    public:
        static void add_histogram( const Mat& histogram,
                                   const Scalar& color,
                                   Mat& image );
    private:
        Utility() {}
    };
}