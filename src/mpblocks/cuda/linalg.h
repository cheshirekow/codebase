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
 *  @file   mpblocks/cuda/linalg.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG_H_
#define MPBLOCKS_CUDA_LINALG_H_


namespace mpblocks {
namespace cuda     {
namespace linalg   {




} // linalg
} // cuda
} // mpblocks

#include <cassert>
#include <mpblocks/cuda/linalg/StreamAssignment.h>
#include <mpblocks/cuda/linalg/RValue.h>
#include <mpblocks/cuda/linalg/LValue.h>
#include <mpblocks/cuda/linalg/Difference.h>
#include <mpblocks/cuda/linalg/Sum.h>
#include <mpblocks/cuda/linalg/Product.h>
#include <mpblocks/cuda/linalg/Scale.h>
#include <mpblocks/cuda/linalg/Transpose.h>
#include <mpblocks/cuda/linalg/Normalize.h>
#include <mpblocks/cuda/linalg/View.h>
#include <mpblocks/cuda/linalg/Matrix.h>
#include <mpblocks/cuda/linalg/Rotation2d.h>
#include <mpblocks/cuda/linalg/iostream.h>




#endif // LINALG_H_
