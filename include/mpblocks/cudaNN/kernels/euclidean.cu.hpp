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

#ifndef MPBLOCKS_CUDANN_KERNELS_EUCLIDEAN_CU_HPP_
#define MPBLOCKS_CUDANN_KERNELS_EUCLIDEANCU_HPP_

#include <mpblocks/cuda/linalg2.h>
#include <mpblocks/cudaNN/kernels/euclidean.cu.h>
#include <mpblocks/cudaNN/kernels/Reader.cu.hpp>

namespace    mpblocks {
namespace      cudaNN {
namespace     kernels {

namespace linalg = cuda::linalg2;


template< typename Scalar, int NDim >
__device__
Scalar euclidean_pseudo_distance(
        const linalg::Matrix<Scalar,NDim,1>& q0,
        const linalg::Matrix<Scalar,NDim,1>& q1 )
{
    return linalg::norm_squared(q1-q0);
}

template< typename Scalar, int NDim >
__device__
Scalar euclidean_distance(
        const linalg::Matrix<Scalar,NDim,1>& q0,
        const linalg::Matrix<Scalar,NDim,1>& q1 )
{
    return linalg::norm(q1-q0);
}


template< typename Format_t, unsigned int NDim>
__global__  void euclidean_distance(
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
    linalg::Matrix<Format_t,NDim,1> q0,q1;

    // read in the query point q0, no synchronization between reads
    read(q,q0);

    // read in the target point q1, we synchronize between reads so that
    // reads are coallesced for maximum throughput
    read(g_in,pitchIn,idx,q1);

    // now compute the distance for this point
    Format_t d = linalg::norm_squared(q1-q0);
    __syncthreads();
    g_out[0*pitchOut + idx] = d;
    __syncthreads();
    g_out[1*pitchOut + idx] = idx;
}


template <int Arg,int i,bool final>
struct NextPow2_iter
{
    enum{ value = NextPow2_iter<Arg,2*i,(2*i>Arg)>::value };
};

template <int Arg,int i>
struct NextPow2_iter<Arg,i,true>
{
    enum{ value = i };
};

template <int Arg>
struct NextPow2
{
    enum{ value = NextPow2_iter<Arg,1,false>::value };
};


template< bool Pseudo, typename Scalar, unsigned int NDim>
__global__  void euclidean_distance(
       RectangleQuery<Scalar,NDim> query,
       Scalar*    g_out
       )
{
    using namespace linalg;

    __shared__ Scalar med[NDim];
    __shared__ Scalar min[NDim];
    __shared__ Scalar max[NDim];
    __shared__ Scalar dist[NDim];

    int idx           = threadIdx.x;
    int childIdx      = blockIdx.x;          ///< child spec

    // the first NDim threads for this block build up the child hyper
    // rectangle for this block
    if( idx < NDim )
    {
        med[idx] = Scalar(0.5)*( query.min[idx] + query.max[idx] );
    }
    __syncthreads();

    if( idx < NDim )
    {
        if( childIdx & (0x01 << idx ) )
        {
            min[idx] = med[idx];
            max[idx] = query.max[idx];
        }
        else
        {
            min[idx] = query.min[idx];
            max[idx] = med[idx];
        }
    }
    __syncthreads();

    // each thread reads in his data
    Scalar q_i    = query.point[idx];
    __syncthreads();

    Scalar min_i  = min[idx];
    __syncthreads();

    Scalar max_i  = max[idx];
    __syncthreads();

    Scalar dist_i = 0;
    if( q_i < min_i )
        dist_i = min_i - q_i;
    if( q_i > max_i )
        dist_i = q_i - max_i;

    dist_i *= dist_i;
    dist[idx] = dist_i;
    __syncthreads();

    for(int s=NextPow2<NDim>::value; s>0; s>>=1 )
    {
        if(idx < s && idx+s < NDim)
            dist[idx] += dist[idx+s];
        __syncthreads();
    }

    if( idx ==0 )
    {
        if(Pseudo)
            g_out[childIdx] = dist[0];
        else
            g_out[childIdx] = std::sqrt(dist[0]);
    }
}







} // kernels
} // cudaNN
} // mpblocks


#endif

