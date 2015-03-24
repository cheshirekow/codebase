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

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_KERNELS_CU_H_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_KERNELS_CU_H_

#include <cuda.h>
#include <cuda_runtime.h>

namespace    mpblocks {
namespace      dubins {
namespace curves_cuda {
namespace     kernels {


/// batch-compute the distance from a single dubins state to a batch of
/// many dubins states
template< typename Format_t>
__global__  void distance_to_set(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       );


/// batch-compute the distance from a batch of many dubins states
/// to a single dubins state
template< typename Format_t >
__global__  void distance_from_set(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       );


/// batch-compute the distance from a single dubins state to a batch of
/// many dubins states
template< typename Format_t>
__global__  void distance_to_set_with_id(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       );


/// batch-compute the distance from a batch of many dubins states
/// to a single dubins state
template< typename Format_t >
__global__  void distance_from_set_with_id(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       );


/// batch-compute the distance from a single dubins state to a batch of
/// many dubins states
template< typename Format_t>
__global__  void distance_to_set_debug(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       );

/// batch-compute the distance from a batch of many dubins states
/// to a single dubins state
template< typename Format_t >
__global__  void distance_from_set_debug(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       );







/// batch-compute the euclidean distance from a single dubins state
/// to a batch of many dubins states
template< typename Format_t>
__global__  void group_distance_to_set(
       EuclideanParams<Format_t> p,
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       );


/// batch-compute the euclidean distance from a single dubins state
/// to a batch of many dubins states
template< typename Format_t>
__global__  void group_distance_to_set_with_id(
       EuclideanParams<Format_t> p,
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       );


} // kernels
} // curves
} // dubins
} // mpblocks




#endif // KERNELS_H_
