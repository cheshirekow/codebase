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

#ifndef MPBLOCKS_CUDA_NN_RECT_DIST_H_
#define MPBLOCKS_CUDA_NN_RECT_DIST_H_


#include <mpblocks/cudaNN/QueryPoint.h>

namespace  mpblocks {
namespace    cudaNN {




template< bool Pseudo, typename Scalar, unsigned int NDim >
struct so3_distance
{
    Scalar* g_out;
    Scalar  h_out[16];

    so3_distance();
    ~so3_distance();

    void operator()( RectangleQuery<Scalar,NDim> query,
                  Scalar* h_out );

    template< class inserter >
    void operator()( const RectangleQuery<Scalar,NDim>& query, inserter& insert )
    {
        (*this)(query,h_out);
        std::copy(h_out,h_out+16,insert);
    }
};

template< bool Pseudo, typename Scalar, unsigned int NDim >
struct euclidean_distance
{
    Scalar* g_out;
    Scalar  h_out[16];

    euclidean_distance();
    ~euclidean_distance();

    void operator()( RectangleQuery<Scalar,NDim> query,
                  Scalar* h_out );

    template< class inserter >
    void operator()( const RectangleQuery<Scalar,NDim>& query, inserter& insert )
    {
        (*this)(query,h_out);
        std::copy(h_out,h_out+16,insert);
    }
};



} // cudaNN
} // mpblocks


#endif // RECT_DIST_H_
