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

#ifndef MPBLOCKS_CUDA_CERT_KERNELS_CU_H_
#define MPBLOCKS_CUDA_CERT_KERNELS_CU_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpblocks/cuda/linalg2.h>

namespace  mpblocks {
namespace cuda_cert {
namespace   kernels {

using namespace cuda::linalg2;
typedef Matrix<float,3,3> Matrix3f;
typedef Matrix<float,3,1> Vector3f;
typedef unsigned int uint_t;


__global__  void check_cert_dbg(
       float* g_dataV,
       uint_t i0V,
       uint_t nV,
       uint_t pitchV,
       float* g_dataF,
       uint_t i0F,
       uint_t nF,
       uint_t pitchF,
       Matrix3f R0,
       Matrix3f Rv,
       Vector3f T0,
       Vector3f dT,
       float    gamma,
       float    dilate,
       int*     g_out
       ,float*   g_dbg
       );

__global__  void check_cert(
       float* g_dataV,
       uint_t i0V,
       uint_t nV,
       uint_t pitchV,
       float* g_dataF,
       uint_t i0F,
       uint_t nF,
       uint_t pitchF,
       Matrix3f R0,
       Matrix3f Rv,
       Vector3f T0,
       Vector3f dT,
       float    gamma,
       float    dilate,
       int*     g_out);


__global__  void check_cert2_dbg(
       float* g_dataV,
       uint_t i0V,
       uint_t nV,
       uint_t pitchV,
       float* g_dataF,
       uint_t i0F,
       uint_t nF,
       uint_t pitchF,
       Matrix3f R0,
       Matrix3f Rv,
       Vector3f T0,
       Vector3f dT,
       float    gamma,
       int*     g_out
       ,float*   g_dbg
       );

__global__  void check_cert2(
       float* g_dataV,
       uint_t i0V,
       uint_t nV,
       uint_t pitchV,
       float* g_dataF,
       uint_t i0F,
       uint_t nF,
       uint_t pitchF,
       Matrix3f R0,
       Matrix3f Rv,
       Vector3f T0,
       Vector3f dT,
       float    gamma,
       int*     g_out);



} // kernels
} // cudaNN
} // mpblocks




#endif // KERNELS_H_
