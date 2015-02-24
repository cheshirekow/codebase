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
 *  @file   include/mpblocks/polynomial/Sum.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_SUM_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_SUM_H_

#include <mpblocks/cuda/polynomial/Polynomial.h>

namespace   mpblocks {
namespace       cuda {
namespace polynomial {

/// expression template for sum of two matrix expressions
template <typename Scalar,
            class Exp1, class Spec1,
            class Exp2, class Spec2>
struct Sum:
    public RValue<Scalar, Sum<Scalar,Exp1,Spec1,Exp2,Spec2>,
                  typename intlist::make_union<Spec1,Spec2>::result >
{
    Exp1 const& A;
    Exp2 const& B;

    public:
        __host__ __device__
        Sum( Exp1 const& A, Exp2 const& B ):
            A(A),
            B(B)
        {}

        __host__ __device__
        Scalar eval( Scalar x )
        {
            return A.eval(x) + B.eval(x);
        }
};



template <typename Scalar,
            class Exp1, class Spec1,
            class Exp2, class Spec2>
struct get_spec< Sum<Scalar,Exp1,Spec1,Exp2,Spec2> >
{
    typedef typename intlist::make_union<Spec1,Spec2>::result result;
};


template <int idx, typename Scalar,
            class Exp1, class Spec1,
            class Exp2, class Spec2>
__host__ __device__
Scalar get( const Sum<Scalar,Exp1,Spec1,Exp2,Spec2>& sum )
{
    return get<idx>( sum.A ) + get<idx>( sum.B );
}



template <typename Scalar, class Exp1, class Spec1, class Exp2, class Spec2>
__host__ __device__
Sum<Scalar,Exp1,Spec1,Exp2,Spec2> operator+(
        RValue<Scalar,Exp1,Spec1> const& A,
        RValue<Scalar,Exp2,Spec2> const& B )
{
    typedef Sum<Scalar,Exp1,Spec1,Exp2,Spec2> Sum_t;
    return Sum_t(
            static_cast<Exp1 const&>(A),
            static_cast<Exp2 const&>(B));
}




} // polynomial
} // cuda
} // mpblocks





#endif // SUM_H_
