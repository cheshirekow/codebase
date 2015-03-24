/*
 *  \file   bitonicSort.cu
 *
 *  \date   Sep 3, 2011
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <mpblocks/cuda/bitonic.cu.hpp>

using namespace mpblocks::cuda::bitonic;

/// explicit instanciation
template class Sorter<unsigned int, unsigned int>;
template class Sorter<unsigned int>;

