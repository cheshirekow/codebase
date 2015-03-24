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
 *  @file   include/mpblocks/polynomial.h
 *
 *  @date   Jan 15, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_H_

namespace mpblocks {

/// polynomial arithmetic and algorithms with no dynamic memory access
namespace polynomial {


} // polynomial
} // mpblocks

#include <mpblocks/cuda/polynomial/get_spec.h>
#include <mpblocks/cuda/polynomial/Polynomial.h>
#include <mpblocks/cuda/polynomial/ostream.h>
#include <mpblocks/cuda/polynomial/Sum.h>
#include <mpblocks/cuda/polynomial/Difference.h>
#include <mpblocks/cuda/polynomial/Construct.h>
#include <mpblocks/cuda/polynomial/StreamAssignment.h>
#include <mpblocks/cuda/polynomial/polyval.h>
#include <mpblocks/cuda/polynomial/Normalized.h>
#include <mpblocks/cuda/polynomial/Negative.h>
#include <mpblocks/cuda/polynomial/differentiate.h>
#include <mpblocks/cuda/polynomial/Product.h>
#include <mpblocks/cuda/polynomial/Quotient.h>
#include <mpblocks/cuda/polynomial/SturmSequence.h>
#include <mpblocks/cuda/polynomial/Derivative.h>
#include <mpblocks/cuda/polynomial/SturmSequence2.h>
#include <mpblocks/cuda/polynomial/assign.h>
#include <mpblocks/cuda/polynomial/ScalarSum.h>
#include <mpblocks/cuda/polynomial/ScalarProduct.h>















#endif // POLY_H_
