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

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_SCALARPRODUCT_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_SCALARPRODUCT_H_

#include <mpblocks/cuda/polynomial/Polynomial.h>

namespace   mpblocks {
namespace       cuda {
namespace polynomial {

/// expression template for sum of two matrix expressions
template <typename Scalar, class Exp, class Spec>
struct ScalarProduct:
    public RValue<Scalar, ScalarProduct<Scalar,Exp,Spec>, Spec>
{
    Scalar const& a;
    Exp    const& P;

    public:
        __host__ __device__
        ScalarProduct( const Scalar& a, const Exp& P ):
            a(a),
            P(P)
        {}

        __host__ __device__
        Scalar eval( Scalar x )
        {
            return a*P.eval(x);
        }
};



template <typename Scalar, class Exp, class Spec>
struct get_spec< ScalarProduct<Scalar,Exp,Spec> >
{
    typedef Spec result;
};


template <int idx, typename Scalar, class Exp, class Spec >
__host__ __device__
Scalar get( const ScalarProduct<Scalar,Exp,Spec>& sum )
{
    return sum.a * get<idx>(sum.P);
}


template <typename Scalar, class Exp, class Spec>
__host__ __device__ __forceinline__
ScalarProduct<Scalar,Exp,Spec> operator*( const Scalar& a,
        RValue<Scalar,Exp,Spec> const& B )
{
    return ScalarProduct<Scalar,Exp,Spec>(a,
            static_cast<Exp const&>(B));
}

template <typename Scalar, class Exp, class Spec>
__host__ __device__ __forceinline__
ScalarProduct<Scalar,Exp,Spec> operator*(
        RValue<Scalar,Exp,Spec> const& B,
        const Scalar& a)
{
    return ScalarProduct<Scalar,Exp,Spec>(a,
            static_cast<Exp const&>(B));
}




} // polynomial
} // cuda
} // mpblocks





#endif // SUM_H_
