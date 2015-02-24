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
 *  @file   include/mpblocks/dubins/curves_cuda/kernels.h
 *
 *  @date   Jun 13, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDANN_KERNELS_SE3_CU_H_
#define MPBLOCKS_CUDANN_KERNELS_SE3_CU_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpblocks/cudaNN/QueryPoint.h>

namespace  mpblocks {
namespace    cudaNN {
namespace   kernels {


/// weighted distance between elements of se3, points are interpreted as
/// 3 values of position and 4 values of quaternion
template< typename Format_t, unsigned int NDim>
__global__  void se3_distance(
       Format_t     weight,     ///< weight of rotation distance
       QueryPoint<Format_t,NDim>
                    query,      ///< query point data
       Format_t*    g_in,       ///< input data array
       unsigned int pitchIn,    ///< row pitch of input data
       Format_t*    g_out,      ///< output data array
       unsigned int pitchOut,   ///< row pitch of output data
       unsigned int n           ///< number of points in point set
       );


/// weighted distance between elements of se3, points are interpreted as
/// 3 values of position and 4 values of quaternion, in this distance we
/// use the [0,1] non-lenear quaternion distance 1-q*q
template< typename Format_t, unsigned int NDim>
__global__  void se3_pseudo_distance(
       Format_t     weight,     ///< weight of rotation distance
       QueryPoint<Format_t,NDim>
                    query,      ///< query point data
       Format_t*    g_in,       ///< input data array
       unsigned int pitchIn,    ///< row pitch of input data
       Format_t*    g_out,      ///< output data array
       unsigned int pitchOut,   ///< row pitch of output data
       unsigned int n           ///< number of points in point set
       );



} // kernels
} // cudaNN
} // mpblocks




#endif // KERNELS_H_
