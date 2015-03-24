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
 *  @file
 *  @date   Oct 30, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA_H_
#define MPBLOCKS_DUBINS_CURVES_CUDA_H_

#include <mpblocks/cuda.h>
#include <mpblocks/cuda/bitonic.h>
#include <mpblocks/cuda/linalg2.h>
#include <mpblocks/dubins/curves/types.h>
#include <mpblocks/dubins/curves/funcs.h>
#include <mpblocks/dubins/curves/result.h>

namespace      mpblocks {
namespace        dubins {
/// classes for solving dubins curves, optimal primitives for shortest path
/// between two states of a dubins vehicle
namespace  curves_cuda {

namespace linalg = cuda::linalg2;


} // curves
} // dubins
} // mpblocks

#include <mpblocks/dubins/curves_cuda/portable.h>
#include <mpblocks/dubins/curves_cuda/Params.h>
#include <mpblocks/dubins/curves_cuda/intrinsics.h>
#include <mpblocks/dubins/curves_cuda/Query.h>
#include <mpblocks/dubins/curves_cuda/PackedIndex.h>
#include <mpblocks/dubins/curves_cuda/PointSet.h>
#include <mpblocks/dubins/curves_cuda/Solution.h>

#endif  // MPBLOCKS_DUBINS_CURVES_CUDA_H_
