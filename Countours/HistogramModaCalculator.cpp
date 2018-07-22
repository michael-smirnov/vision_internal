#include "HistogramModaCalculator.h"

using namespace std;

namespace vision
{
    HistogramModaCalculator::HistogramModaCalculator( const vector<int>& histogram, 
                                                      int window_size )
    : _window_size( window_size )
    , _histogram( histogram )
    {}

    vector<histogram_moda> HistogramModaCalculator::calculate_modas()
    {
        vector<histogram_moda> modas;

        int half_window = _window_size / 2;

        int i = 0;
        while( i < _histogram.size() )
        {
            if( _histogram[i] == 0 )
                i++;
            else
            {
                i = calculate_moda_index_in_growing_argument_order( i );
                i = calculate_moda_by_center_argument( i );

                int moda_sum = 0;
                for( int k = std::max( 0, i - half_window ); 
                     k < std::min( (int)_histogram.size(), i + half_window + 1 ); 
                     k++ )
                {
                    moda_sum += _histogram[k];
                }

                modas.push_back( { i, moda_sum } );
                i += half_window + 1;
            }
        }

        return modas;
    }

    int HistogramModaCalculator::calculate_moda_index_in_growing_argument_order( int start_argument )
    {
        int current_argument = start_argument;

        int half_window = _window_size / 2;

        int sum = _histogram[current_argument];
        for( int j = current_argument + 1; j < _histogram.size() && j < current_argument + 1 + half_window; j++ )
            sum += _histogram[j];

        int k = _histogram[current_argument];
        while( k < sum / 2 )
        {
            current_argument++;
            k += _histogram[current_argument];
        }

        return current_argument;
    }

    int HistogramModaCalculator::calculate_moda_by_center_argument( int center_argument )
    {
        int last_histogram_index = _histogram.size() - 1;
        int half_window = _window_size / 2;

        int start_argument = max( 0, center_argument - half_window );
        int end_index = min( last_histogram_index, center_argument + half_window );

        int sum = 0;
        for( int i = start_argument; i <= end_index; i++ )
            sum += _histogram[i];

        if( sum == 1 )
            return center_argument;

        int current_argument = start_argument;

        int k = _histogram[current_argument];
        while( k < sum / 2 )
        {
            current_argument++;
            k += _histogram[current_argument];
        }

        return current_argument;
    }
}