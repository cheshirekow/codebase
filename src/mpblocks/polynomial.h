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

#ifndef MPBLOCKS_POLYNOMIAL_H_
#define MPBLOCKS_POLYNOMIAL_H_

namespace mpblocks {

/// polynomial arithmetic and algorithms
namespace polynomial {

static const int Dynamic = -0x01;
static const int Sparse  = -0x02;


/// signum
template <typename T> int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}


} // polynomial
} // mpblocks

#include <cassert>
#include <cmath>
#include <mpblocks/linalg.h>
#include <mpblocks/polynomial/Min.h>
#include <mpblocks/polynomial/Max.h>
#include <mpblocks/polynomial/StreamAssignment.h>
#include <mpblocks/polynomial/RValue.h>
#include <mpblocks/polynomial/LValue.h>
#include <mpblocks/polynomial/Polynomial.h>
#include <mpblocks/polynomial/SparsePolynomial.h>
#include <mpblocks/polynomial/polyval.h>
#include <mpblocks/polynomial/differentiate.h>
#include <mpblocks/polynomial/Sum.h>
#include <mpblocks/polynomial/Difference.h>
#include <mpblocks/polynomial/Product.h>
#include <mpblocks/polynomial/Quotient.h>
#include <mpblocks/polynomial/Normalized.h>
#include <mpblocks/polynomial/Negative.h>
#include <mpblocks/polynomial/SturmSequence.h>
#include <mpblocks/polynomial/ostream.h>















#endif // POLY_H_
