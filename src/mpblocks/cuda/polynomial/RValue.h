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
 *  @file   include/mpblocks/polynomial/RValue.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_RVALUE_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_RVALUE_H_

namespace   mpblocks {
namespace       cuda {
namespace polynomial {

template< int idx, typename Scalar, class Exp1 >
__host__ __device__
Scalar get( const Exp1& );

/// expression template for rvalues
template <class Scalar, class Exp, class Spec>
struct RValue
{};


template <typename Scalar, class Exp, class Spec>
__host__ __device__
RValue<Scalar,Exp,Spec>& rvalue( RValue<Scalar,Exp,Spec>& exp )
{
    return exp;
}





} // polynomial
} // cuda
} // mpblocks





#endif // MATRIXEXPRESSION_H_
