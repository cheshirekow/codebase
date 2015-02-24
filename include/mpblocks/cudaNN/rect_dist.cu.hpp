/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of mpblocks.
 *
 *  mpblocks is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  mpblocks is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with mpblocks.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_nn/include/mpblocks/cudaNN/rect_dist.h
 *
 *  @date   Nov 13, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_NN_RECT_DIST_CU_HPP_
#define MPBLOCKS_CUDA_NN_RECT_DIST_CU_HPP_

#include <mpblocks/cuda.hpp>
#include <mpblocks/cudaNN/rect_dist.h>
#include <mpblocks/cudaNN/kernels/so3.cu.hpp>
#include <mpblocks/cudaNN/kernels/euclidean.cu.hpp>

namespace  mpblocks {
namespace    cudaNN {

template< bool Pseudo, typename Scalar, unsigned int NDim >
so3_distance<Pseudo,Scalar,NDim>::so3_distance()
{
    g_out = cuda::mallocT<Scalar>(16);
}

template< bool Pseudo, typename Scalar, unsigned int NDim >
so3_distance<Pseudo,Scalar,NDim>::~so3_distance()
{
    cuda::free(g_out);
}

template< bool Pseudo, typename Scalar, unsigned int NDim >
void so3_distance<Pseudo,Scalar,NDim>::operator()(
       RectangleQuery<Scalar,NDim>  query,
       Scalar*                      h_out
       )
{
    int threads = 192;
    int blocks  = 16;

    kernels::so3_distance<Pseudo,Scalar,NDim><<<blocks,threads>>>(query,g_out);
    cuda::deviceSynchronize();
    cuda::memcpyT<float>(h_out,g_out,16,cudaMemcpyDeviceToHost);
}


template< bool Pseudo, typename Scalar, unsigned int NDim >
euclidean_distance<Pseudo,Scalar,NDim>::euclidean_distance()
{
    g_out = cuda::mallocT<Scalar>(16);
}

template< bool Pseudo, typename Scalar, unsigned int NDim >
euclidean_distance<Pseudo,Scalar,NDim>::~euclidean_distance()
{
    cuda::free(g_out);
}

template< bool Pseudo, typename Scalar, unsigned int NDim >
void euclidean_distance<Pseudo,Scalar,NDim>::operator()(
       RectangleQuery<Scalar,NDim>  query,
       Scalar*                      h_out
       )
{
    int threads = NDim;
    int blocks  = 0x01 << NDim;

    kernels::euclidean_distance<Pseudo,Scalar,NDim><<<blocks,threads>>>(query,g_out);
    cuda::deviceSynchronize();
    cuda::memcpyT<float>(h_out,g_out,blocks,cudaMemcpyDeviceToHost);
}



} // cudaNN
} // mpblocks


#endif // RECT_DIST_H_
