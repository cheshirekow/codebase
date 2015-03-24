/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of openbook.
 *
 *  openbook is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  openbook is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with openbook.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   dubins/curves_cuda/kernels.cu
 *
 *  @date   Jun 13, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDANN_KERNELS_R2S1_CU_HPP_
#define MPBLOCKS_CUDANN_KERNELS_R2S1_CU_HPP_

#include <mpblocks/cuda/linalg2.h>
#include <mpblocks/cudaNN/kernels/r2s1.cu.h>
#include <mpblocks/cudaNN/kernels/Reader.cu.hpp>

namespace    mpblocks {
namespace      cudaNN {
namespace     kernels {

namespace linalg = cuda::linalg2;




template< typename Format_t, unsigned int NDim>
__global__  void r2s1_distance(
       Format_t     weight,
       QueryPoint<Format_t,NDim> q,
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       )
{
    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int N        = blockDim.x;  ///< number of blocks

    // which data point we work on
    int idx      = blockId * N + threadId;

    // if our idx is greater than the number of data points then we are a
    // left-over thread so just bail
    // @todo is this OK with non-power of
    // two array sizes and the fact that we syncthreads after this point?
    if( idx > n )
        return;

    // compose the query object
    linalg::Matrix<Format_t,2,1> x0,x1;
    Format_t                     t0,t1;

    // read in the query point q0, no synchronization between reads
    linalg::set<0>(x0) = q.data[0];
    linalg::set<1>(x0) = q.data[1];
    t0                 = q.data[2];
    linalg::set<0>(x1) = g_in[idx + 0*pitchIn]; __syncthreads();
    linalg::set<1>(x1) = g_in[idx + 1*pitchIn]; __syncthreads();
    t1                 = g_in[idx + 2*pitchIn]; __syncthreads();

    // now compute the distance for this point
    Format_t da = __fsqrt_rn(linalg::norm_squared(x1-x0))
                    + weight*( t1 - t0 );
    Format_t db = __fsqrt_rn(linalg::norm_squared(x1-x0))
                    + weight*( 2*M_PI - (t1 - t0) );
    Format_t d  = fminf(da,db);
    __syncthreads();
    g_out[0*pitchOut + idx] = d;
    __syncthreads();
    g_out[1*pitchOut + idx] = idx;
}





} // kernels
} // cudaNN
} // mpblocks


#endif

