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

#ifndef MPBLOCKS_CUDANN_KERNELS_SE3_CU_HPP_
#define MPBLOCKS_CUDANN_KERNELS_SE3_CU_HPP_

#include <mpblocks/cuda/linalg2.h>
#include <mpblocks/cudaNN/kernels/se3.cu.h>
#include <mpblocks/cudaNN/kernels/Reader.cu.hpp>

namespace    mpblocks {
namespace      cudaNN {
namespace     kernels {

namespace linalg = cuda::linalg2;


template< typename Format_t, unsigned int NDim>
__global__  void se3_pseudo_distance(
       Format_t     weight,
       QueryPoint<Format_t,NDim> q,
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       )
{
    using namespace linalg;

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
    linalg::Matrix<Format_t,3,1>  t0,t1;
    linalg::Matrix<Format_t,4,1>  q0,q1;

    // read in the query point q0, no synchronization between reads
    if( NDim >= 7 )
    {
        set<0>(t0) = q.data[0];
        set<1>(t0) = q.data[1];
        set<2>(t0) = q.data[2];
        set<0>(q0) = q.data[3];
        set<1>(q0) = q.data[4];
        set<2>(q0) = q.data[5];
        set<3>(q0) = q.data[6];

        set<0>(t1) = g_in[0*pitchIn + idx];
        __syncthreads();
        set<1>(t1) = g_in[1*pitchIn + idx];
        __syncthreads();
        set<2>(t1) = g_in[2*pitchIn + idx];
        __syncthreads();
        set<0>(q1) = g_in[3*pitchIn + idx];
        __syncthreads();
        set<1>(q1) = g_in[4*pitchIn + idx];
        __syncthreads();
        set<2>(q1) = g_in[5*pitchIn + idx];
        __syncthreads();
        set<3>(q1) = g_in[6*pitchIn + idx];
        __syncthreads();

        // now compute the distance for this point
        Format_t dq = linalg::dot(q0,q1);
        Format_t d  = linalg::norm_squared(t1-t0) + weight*(1-dq*dq);
        __syncthreads();
        g_out[0*pitchOut + idx] = d;
        __syncthreads();
        g_out[1*pitchOut + idx] = idx;
    }
}



template< typename Format_t, unsigned int NDim>
__global__  void se3_distance(
       Format_t     weight,
       QueryPoint<Format_t,NDim> q,
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       )
{
    using namespace linalg;

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
    linalg::Matrix<Format_t,3,1>  t0,t1;
    linalg::Matrix<Format_t,4,1>  q0,q1;

    // read in the query point q0, no synchronization between reads
    if( NDim >= 7 )
    {
        set<0>(t0) = q.data[0];
        set<1>(t0) = q.data[1];
        set<2>(t0) = q.data[2];
        set<0>(q0) = q.data[3];
        set<1>(q0) = q.data[4];
        set<2>(q0) = q.data[5];
        set<3>(q0) = q.data[6];

        set<0>(t1) = g_in[0*pitchIn + idx];
        __syncthreads();
        set<1>(t1) = g_in[1*pitchIn + idx];
        __syncthreads();
        set<2>(t1) = g_in[2*pitchIn + idx];
        __syncthreads();
        set<0>(q1) = g_in[3*pitchIn + idx];
        __syncthreads();
        set<1>(q1) = g_in[4*pitchIn + idx];
        __syncthreads();
        set<2>(q1) = g_in[5*pitchIn + idx];
        __syncthreads();
        set<3>(q1) = g_in[6*pitchIn + idx];
        __syncthreads();

        // now compute the distance for this point
        Format_t dq  = linalg::dot(q0,q1);
        Format_t arg = 2*dq*dq - 1;
        arg = fmaxf(-0.999999999999999999999999999f,
              fminf(arg, 0.9999999999999999999999999f));
        Format_t d  =
                sqrtf(linalg::norm_squared(t1-t0)) + weight*acosf( arg );
        __syncthreads();
        g_out[0*pitchOut + idx] = d;
        __syncthreads();
        g_out[1*pitchOut + idx] = idx;
    }
}













} // kernels
} // cudaNN
} // mpblocks


#endif

