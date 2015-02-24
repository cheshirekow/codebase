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
 *  @file   /home/josh/Codes/cpp/mpblocks2/dubins/include/mpblocks/dubins/curves_cuda/intrinsics.h
 *
 *  @date   Jun 14, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA_INTRINSICS_H_
#define MPBLOCKS_DUBINS_CURVES_CUDA_INTRINSICS_H_

#ifdef __CUDACC__
#ifdef __CUDA_ARCH__
#define WHICH_CUDA intr::DEVICE
#else
#define WHICH_CUDA intr::HOST
#endif
#else
#define WHICH_CUDA intr::NATIVE
#endif

#include <cuda_runtime.h>
#include <mpblocks/cuda/linalg2.h>

namespace     mpblocks {
namespace       dubins {
namespace  curves_cuda {

// provides overloads for cuda intrinsics for generic programming
namespace         intr {

enum Trajectory { DEVICE, HOST, NATIVE };

// defeault template calls std functions
template <Trajectory>
struct Dispatch {
  __host__ static double sin(double x)  { return std::sin(x); }
  __host__ static float  sin(float x)   { return std::sin(x); }
  __host__ static double cos(double x)  { return std::cos(x); }
  __host__ static float  cos(float x)   { return std::cos(x); }
  __host__ static double acos(double x) { return std::acos(x); }
  __host__ static float  acos(float x)  { return std::acos(x); }
  __host__ static double atan2(double x, double y) { return std::atan2(x, y); }
  __host__ static float  atan2(float x,  float  y) { return std::atan2(x, y); }
};

// specialization for device calls device functions
template <>
struct Dispatch<DEVICE> {
  __device__ static double sin(double x)  { return ::sin(x); }
  __device__ static float  sin(float x)   { return __sinf(x); }
  __device__ static double cos(double x)  { return ::cos(x); }
  __device__ static float  cos(float x)   { return __cosf(x); }
  __device__ static double acos(double x) { return ::acos(x); }
  __device__ static float  acos(float x)  { return ::acosf(x); }
  __device__ static double atan2(double x, double y) { return ::atan2(x, y); }
  __device__ static float  atan2(float x, double y)  { return ::atan2f(x, y);}
};

} // intrinsics
} // curves
} // dubins
} // mpblocks

#endif // INTRINSICS_H_
