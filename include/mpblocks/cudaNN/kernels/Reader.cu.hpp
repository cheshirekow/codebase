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

#ifndef MPBLOCKS_CUDANN_KERNELS_READER_CU_HPP_
#define MPBLOCKS_CUDANN_KERNELS_READER_CU_HPP_

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpblocks/cuda/linalg2.h>

namespace    mpblocks {
namespace      cudaNN {
namespace     kernels {

namespace linalg = cuda::linalg2;

// recursive implentation of assignment iteration
template <typename Format_t, unsigned int NDim, unsigned int Idx>
struct Reader
{
    __device__ inline static
    void read( Format_t* g_data, unsigned int pitch, int idx,
                     linalg::Matrix<Format_t,NDim,1>& v )
    {
        linalg::set<Idx>(v) = g_data[Idx*pitch + idx];
        __syncthreads();
        Reader<Format_t,NDim,Idx+1>::read(g_data,pitch,idx,v);
    }

    __device__ inline static
    void read( const QueryPoint<Format_t,NDim>& q,
               linalg::Matrix<Format_t,NDim,1>& v )
    {
        linalg::set<Idx>(v) = q.data[Idx];
        __syncthreads();
        Reader<Format_t,NDim,Idx+1>::read(q,v);
    }
};

template <typename Format_t, unsigned int NDim>
struct Reader<Format_t,NDim,NDim>
{
    __device__ inline static
    void read( Format_t* g_data, unsigned int pitch, int idx,
                     linalg::Matrix<Format_t,NDim,1>& v )
    {}

    __device__ inline static
    void read( const QueryPoint<Format_t,NDim>& q,
               linalg::Matrix<Format_t,NDim,1>& v )
    {}
};

template <typename Format_t, unsigned int NDim>
__device__ inline
void read( Format_t* g_data, unsigned int pitch, int idx,
                 linalg::Matrix<Format_t,NDim,1>& v )
{
    Reader<Format_t,NDim,0>::read(g_data,pitch,idx,v);
}

template <typename Format_t, unsigned int NDim>
__device__ inline
void read( const QueryPoint<Format_t,NDim>&q,
                 linalg::Matrix<Format_t,NDim,1>& v )
{
    Reader<Format_t,NDim,0>::read(q,v);
}













} // kernels
} // cudaNN
} // mpblocks


#endif

