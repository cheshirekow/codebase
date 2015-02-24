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

#ifndef MPBLOCKS_CUDANN_KERNELS_CU_H_
#define MPBLOCKS_CUDANN_KERNELS_CU_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <mpblocks/cudaNN/QueryPoint.h>
#include <mpblocks/cudaNN/kernels/euclidean.cu.h>
#include <mpblocks/cudaNN/kernels/r2s1.cu.h>
#include <mpblocks/cudaNN/kernels/se3.cu.h>
#include <mpblocks/cudaNN/kernels/so3.cu.h>

namespace  mpblocks {
namespace    cudaNN {
namespace   kernels {


} // kernels
} // cudaNN
} // mpblocks




#endif // KERNELS_H_
