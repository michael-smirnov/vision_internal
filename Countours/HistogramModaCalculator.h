#pragma once

#include "types.h"

#include <vector>

namespace vision
{
    struct histogram_moda
    {
        int argument;
        int values_count;
    };

    class HistogramModaCalculator
    {
    public:
        HistogramModaCalculator( const std::vector<int>& histogram, 
                                 int window_size );
        std::vector<histogram_moda> calculate_modas();

    private:
        int calculate_moda_index_in_growing_argument_order( int start_argument );
        int calculate_moda_by_center_argument( int center_argument );

    private:
        const int _window_size;
        std::vector<int> _histogram;
    };
}