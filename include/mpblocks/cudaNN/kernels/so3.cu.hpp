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

#ifndef MPBLOCKS_CUDANN_KERNELS_SO3_CU_HPP_
#define MPBLOCKS_CUDANN_KERNELS_SO3_CU_HPP_

#include <mpblocks/cuda/linalg2.h>
#include <mpblocks/cudaNN/kernels/so3.cu.h>
#include <mpblocks/cudaNN/kernels/Reader.cu.hpp>

namespace    mpblocks {
namespace      cudaNN {
namespace     kernels {

namespace linalg = cuda::linalg2;

template< typename Scalar >
__device__
Scalar so3_pseudo_distance(
        const linalg::Matrix<Scalar,4,1>& q0,
        const linalg::Matrix<Scalar,4,1>& q1 )
{
    Scalar dq = linalg::dot(q0,q1);
    dq = fmaxf(-1,fminf(dq,1));
    return 1-dq;
}

template< typename Scalar >
__device__
Scalar so3_distance(
        const linalg::Matrix<Scalar,4,1>& q0,
        const linalg::Matrix<Scalar,4,1>& q1 )
{
    Scalar dot = linalg::dot(q0,q1);
    Scalar arg = 2*dot*dot - 1;
    arg = fmaxf(-0.999999999999999999999999999f,
              fminf(arg, 0.9999999999999999999999999f));
    return acosf(arg);
}

template< typename Scalar, bool Pseudo >
struct so3_distance_fn
{
    __device__
    static Scalar compute(
            const linalg::Matrix<Scalar,4,1>& q0,
            const linalg::Matrix<Scalar,4,1>& q1 )
    {
        return so3_distance(q0,q1);
    }
};

template< typename Scalar>
struct so3_distance_fn<Scalar,true>
{
    __device__
    static Scalar compute(
            const linalg::Matrix<Scalar,4,1>& q0,
            const linalg::Matrix<Scalar,4,1>& q1 )
    {
        return so3_pseudo_distance(q0,q1);
    }
};


template< bool Pseudo, typename Scalar, unsigned int NDim>
__global__  void so3_distance(
       QueryPoint<Scalar,NDim> q,
       Scalar*    g_in,
       unsigned int pitchIn,
       Scalar*    g_out,
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
    linalg::Matrix<Scalar,4,1>  q0,q1;

    // read in the query point q0, no synchronization between reads
    if( NDim >= 4 )
    {
        set<0>(q0) = q.data[0];
        set<1>(q0) = q.data[1];
        set<2>(q0) = q.data[2];
        set<3>(q0) = q.data[3];

        set<0>(q1) = g_in[0*pitchIn + idx];
        __syncthreads();
        set<1>(q1) = g_in[1*pitchIn + idx];
        __syncthreads();
        set<2>(q1) = g_in[2*pitchIn + idx];
        __syncthreads();
        set<3>(q1) = g_in[3*pitchIn + idx];
        __syncthreads();

        // now compute the distance for this point
        Scalar d  = so3_distance_fn<Scalar,Pseudo>::compute(q0,q1);
        __syncthreads();
        g_out[0*pitchOut + idx] = d;
        __syncthreads();
        g_out[1*pitchOut + idx] = idx;
    }
}

enum Constraint
{
    OFF = 0,
    MIN = 1,
    MAX = 2
};



template< bool Pseudo, typename Scalar, unsigned int NDim>
__global__  void so3_distance(
       RectangleQuery<Scalar,NDim> query,
       Scalar*    g_out
       )
{
    using namespace linalg;

    __shared__ Scalar med[4];
    __shared__ Scalar min[4];
    __shared__ Scalar max[4];
    __shared__ Scalar dist[192];

    int idx           = threadIdx.x;
    int constraintIdx = idx % 81;    ///< constraint spec
    int sign          = idx > 80 ? -1 : 1;
    int childIdx      = blockIdx.x;          ///< child spec

    // all threads initialize their output block
    dist[idx] = 1000;
    __syncthreads();

    // the first four threads for this block build up the child hyper
    // rectangle for this block
    if( idx < 4 )
    {
        med[idx] = Scalar(0.5)*( query.min[idx] + query.max[idx] );

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

    // if our idx is greater than the number of output points then we are a
    // left-over thread so just bail
    if( idx >= 81*2  )
        return;

    // compose the query matrices
    linalg::Matrix<Scalar,4,1>  q,x;

    // the if statement just avoids useless cuda warnings
    if( NDim >= 4 )
    {
        set<0>(q) = query.point[0];
        set<1>(q) = query.point[1];
        set<2>(q) = query.point[2];
        set<3>(q) = query.point[3];

        // compute the specification for this thread
        int  cnst = constraintIdx;;
        char spec_0 = cnst % 3; cnst /= 3;
        char spec_1 = cnst % 3; cnst /= 3;
        char spec_2 = cnst % 3; cnst /= 3;
        char spec_3 = cnst % 3; cnst /= 3;

        // build up the lambda calculation for this thread
        Scalar den = 1;
        Scalar num = 0;

        if( spec_0 == MAX )
            den -= max[0]*max[0];
        else if( spec_0 == MIN )
            den -= min[0]*min[0];
        else
            num += get<0>(q)*get<0>(q);

        if( spec_1 == MAX )
            den -= max[1]*max[1];
        else if( spec_1 == MIN )
            den -= min[1]*min[1];
        else
            num += get<1>(q)*get<1>(q);

        if( spec_2 == MAX )
            den -= max[2]*max[2];
        else if( spec_2 == MIN )
            den -= min[2]*min[2];
        else
            num += get<2>(q)*get<2>(q);

        if( spec_3 == MAX )
            den -= max[3]*max[3];
        else if( spec_3 == MIN )
            den -= min[3]*min[3];
        else
            num += get<3>(q)*get<3>(q);

        bool feasible     = true;

        if( den < 1e-8 )
            feasible = false;

        Scalar lambda2  = num / (4*den);
        Scalar lambda   = std::sqrt(lambda2);

        if( idx == 0 )
        {
            lambda = -Scalar(1.0)/(Scalar(2.0)*sign);
            feasible  = true;
        }

        linalg::Matrix<Scalar,4,1> x;

        if( spec_0 == MAX )
            set<0>(x) = max[0];
        else if( spec_0 == MIN )
            set<0>(x) = min[0];
        else
            set<0>(x) = -get<0>(q) / (2*sign*lambda);

        if( get<0>(x) > max[0] || get<0>(x) < min[0] )
            feasible = false;

        if( spec_1 == MAX )
            set<1>(x) = max[1];
        else if( spec_1 == MIN )
            set<1>(x) = min[1];
        else
            set<1>(x) = -get<1>(q) / (2*sign*lambda);

        if( get<1>(x) > max[1] || get<1>(x) < min[1] )
            feasible = false;

        if( spec_2 == MAX )
            set<2>(x) = max[2];
        else if( spec_2 == MIN )
            set<2>(x) = min[2];
        else
            set<2>(x) = -get<2>(q) / (2*sign*lambda);

        if( get<2>(x) > max[2] || get<2>(x) < min[2] )
            feasible = false;

        if( spec_3 == MAX )
            set<3>(x) = max[3];
        else if( spec_3 == MIN )
            set<3>(x) = min[3];
        else
            set<3>(x) = -get<3>(q) / (2*sign*lambda);

        if( get<3>(x) > max[3] || get<3>(x) < min[3] )
            feasible = false;

        // now compute the distance for this point
        Scalar d  = so3_distance_fn<Scalar,Pseudo>::compute(q,x);
        if(idx == 0 )
                d = 0;
        if(!feasible)
                d = 1000;

        dist[idx] = d;
        __syncthreads();

        // only the first warp does the reduction
        if( idx < 32 )
        {
            // start by taking the min across warps
            Scalar dist_i = dist[idx];
            Scalar dist_j = 1000;

            for(int j=1; j < 6; j++)
            {
                Scalar dist_j = dist[idx + j*32];
                if( dist_j < dist_i )
                    dist_i = dist_j;
            }

            // write out our min, no need to sync b/c warp synchronous
            dist[idx] = dist_i;

            // now perform the reduction
            for(int j=16; j > 0; j /= 2 )
            {
                dist_j = dist[idx + j];
                if( dist_j < dist_i )
                    dist_i = dist_j;
                dist[idx] = dist_i;
            }

            if( idx ==0 )
                g_out[childIdx] = dist_i;
        }
    }
}












} // kernels
} // cudaNN
} // mpblocks


#endif

