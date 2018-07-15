#pragma once

#include "types.h"
#include <vector>

namespace vision
{
    class MathItemGenerator
    {
    public:
        static Mat matrix_by_row( const std::vector<float>& row );
        static Mat color_map_matrix( const Mat& m, int16_t min_value, int16_t max_value );
    private:
        MathItemGenerator() {}
    };
}