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
 *  @file   pblocks/polynomial/LValue.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_LVALUE_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_LVALUE_H_

#include <mpblocks/cuda/polynomial/RValue.h>

namespace   mpblocks {
namespace       cuda {
namespace polynomial {


template< int idx, typename Scalar, class Exp1 >
__host__ __device__
Scalar& set( Exp1& );




template <typename Scalar, class Exp1, class Exp2, class Spec>
struct AssignmentHelper{};

template <typename Scalar, class Exp1, class Exp2, int Head, class Tail>
struct AssignmentHelper< Scalar, Exp1, Exp2, IntList<Head,Tail> >
{
    __host__ __device__
    static void assign( Exp1& lvalue, const Exp2& rvalue )
    {
        set<Head>( lvalue ) = get<Head>( rvalue );
        AssignmentHelper< Scalar, Exp1, Exp2, Tail >::assign(lvalue,rvalue);
    }
};

template <typename Scalar, class Exp1, class Exp2, int Tail>
struct AssignmentHelper< Scalar, Exp1, Exp2, IntList<Tail,intlist::Terminal> >
{
    __host__ __device__
    static void assign( Exp1& lvalue, const Exp2& rvalue )
    {
        set<Tail>( lvalue ) = get<Tail>( rvalue );
    }
};


/// intermediate object which allows LValue assignment operator to call
/// differntiate
/// expression template for sum of two matrix expressions
template <int n, typename Scalar, class Exp, class Spec>
struct DerivativeSurrogate
{
    Exp const& exp;

    __host__ __device__
    DerivativeSurrogate( Exp const& exp ):
        exp(exp)
    {}
};


template <typename Scalar, class Exp>
class LValue;

template <int n, typename Scalar, class Exp1, class InSpec, class Exp2>
__host__ __device__
void differentiate( const RValue<Scalar,Exp1,InSpec>& in,
                    LValue<Scalar,Exp2>& out );


/// expression template for lvalues
template <typename Scalar, class Exp>
class LValue
{
    public:
//        template <class Exp2, class Spec>
//        __host__ __device__
//        LValue<Scalar,Exp>& operator=( RValue<Scalar,Exp2,Spec> const& rvalue )
//        {
//            AssignmentHelper<Scalar,Exp,Exp2,Spec>::assign(
//                    static_cast<Exp&>(*this),
//                    static_cast<const Exp2&>( rvalue ) );
//            return *this;
//        }

        template <int n, class Exp2, class Spec2>
        __host__ __device__
        LValue<Scalar,Exp>& operator=(
                DerivativeSurrogate<n,Scalar,Exp2,Spec2>& surrogate )
        {
            differentiate<n>( surrogate.exp, *this );
            return *this;
        }
};




template <typename Scalar, class Exp>
__host__ __device__
LValue<Scalar,Exp>& lvalue( LValue<Scalar,Exp>& exp )
{
    return exp;
}








} // polynomial
} // cuda
} // mpblocks





#endif // MATRIXEXPRESSION_H_
