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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_nn/src/kernels.cu
 *
 *  @date   Oct 6, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <mpblocks/cudaNN/kernels.cu.hpp>



namespace  mpblocks {
namespace    cudaNN {

template struct QueryPoint<float,2>;
template struct QueryPoint<float,3>;
template struct QueryPoint<float,7>;

namespace   kernels {

template __global__  void euclidean_distance<float,2>(
       QueryPoint<float,2>
                    query,
       float*       g_in,
       unsigned int pitchIn,
       float*       g_out,
       unsigned int pitchOut,
       unsigned int n
       );

template __global__  void euclidean_distance<float,3>(
       QueryPoint<float,3>
                    query,
       float*       g_in,
       unsigned int pitchIn,
       float*       g_out,
       unsigned int pitchOut,
       unsigned int n
       );

template __global__  void r2s1_distance<float,3>(
       float        weight,
       QueryPoint<float,3>
                    query,
       float*       g_in,
       unsigned int pitchIn,
       float*       g_out,
       unsigned int pitchOut,
       unsigned int n
       );

template __global__  void euclidean_distance<float,7>(
       QueryPoint<float,7>
                    query,
       float*       g_in,
       unsigned int pitchIn,
       float*       g_out,
       unsigned int pitchOut,
       unsigned int n
       );


template __global__  void se3_distance<float,7>(
       float        weight,     ///< weight of rotation distance
       QueryPoint<float,7>
                    query,      ///< query point data
       float*       g_in,       ///< input data array
       unsigned int pitchIn,    ///< row pitch of input data
       float*       g_out,      ///< output data array
       unsigned int pitchOut,   ///< row pitch of output data
       unsigned int n           ///< number of points in point set
       );


template __global__  void so3_distance<false,float,4>(
       QueryPoint<float,4>
                    query,      ///< query point data
       float*       g_in,       ///< input data array
       unsigned int pitchIn,    ///< row pitch of input data
       float*       g_out,      ///< output data array
       unsigned int pitchOut,   ///< row pitch of output data
       unsigned int n           ///< number of points in point set
       );

template __global__  void so3_distance<true,float,4>(
       QueryPoint<float,4>
                    query,      ///< query point data
       float*       g_in,       ///< input data array
       unsigned int pitchIn,    ///< row pitch of input data
       float*       g_out,      ///< output data array
       unsigned int pitchOut,   ///< row pitch of output data
       unsigned int n           ///< number of points in point set
       );

template __global__  void so3_distance<false,float,4>(
       RectangleQuery<float,4>
                    query,      ///< query point data
       float*       g_out       ///< output data array
       );

template __global__  void so3_distance<true,float,4>(
       RectangleQuery<float,4>
                    query,      ///< query point data
       float*       g_out       ///< output data array
       );

} // kernels
} // cudaNN
} // mpblocks





