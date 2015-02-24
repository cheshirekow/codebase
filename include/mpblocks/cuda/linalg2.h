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
 *  @file   mpblocks/cuda/linalg2.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_H_
#define MPBLOCKS_CUDA_LINALG2_H_


namespace mpblocks {
namespace cuda     {
namespace linalg2  {

typedef unsigned int Size_t;


} // linalg
} // cuda
} // mpblocks

#include <cassert>
#include <mpblocks/cuda/linalg2/Assignment.h>
#include <mpblocks/cuda/linalg2/AssignmentIterator.h>
#include <mpblocks/cuda/linalg2/RValue.h>
#include <mpblocks/cuda/linalg2/LValue.h>
#include <mpblocks/cuda/linalg2/Access.h>
#include <mpblocks/cuda/linalg2/Difference.h>
#include <mpblocks/cuda/linalg2/Sum.h>
#include <mpblocks/cuda/linalg2/Product.h>
#include <mpblocks/cuda/linalg2/Scale.h>
#include <mpblocks/cuda/linalg2/Transpose.h>
//#include <mpblocks/cuda/linalg2/View.h>
#include <mpblocks/cuda/linalg2/Matrix.h>
#include <mpblocks/cuda/linalg2/Rotation2d.h>
#include <mpblocks/cuda/linalg2/View.h>
#include <mpblocks/cuda/linalg2/Norm.h>
#include <mpblocks/cuda/linalg2/Dot.h>
#include <mpblocks/cuda/linalg2/iostream.h>
#include <mpblocks/cuda/linalg2/mktmp.h>





#endif // LINALG2_H_
