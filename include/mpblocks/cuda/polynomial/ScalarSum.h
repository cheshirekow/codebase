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

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_SCALARSUM_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_SCALARSUM_H_

#include <mpblocks/cuda/polynomial/Polynomial.h>

namespace   mpblocks {
namespace       cuda {
namespace polynomial {

/// expression template for sum of two matrix expressions
template <typename Scalar, class Exp, class Spec>
struct ScalarSum:
    public RValue<Scalar, ScalarSum<Scalar,Exp,Spec>,
    typename intlist::make_union< IntList<0,intlist::Terminal>, Spec >::result>
{
    Scalar const& a;
    Exp    const& P;

    public:
        __host__ __device__
        ScalarSum( const Scalar& a, const Exp& P ):
            a(a),
            P(P)
        {}

        __host__ __device__
        Scalar eval( Scalar x )
        {
            return a + P.eval(x);
        }
};



template <typename Scalar, class Exp, class Spec>
struct get_spec< ScalarSum<Scalar,Exp,Spec> >
{
    typedef typename
    intlist::make_union< IntList<0,intlist::Terminal>, Spec >::result result;
};


namespace scalarsum_detail
{
    template <int idx, typename Scalar, class Exp, class Spec >
    struct Get
    {
        __host__ __device__ __forceinline__
        static Scalar do_it( const ScalarSum<Scalar,Exp,Spec>& sum )
        {
            return get<idx>( sum.P );
        }
    };

    template <typename Scalar, class Exp, class Spec >
    struct Get<0,Scalar,Exp,Spec>
    {
        __host__ __device__ __forceinline__
        static Scalar do_it( const ScalarSum<Scalar,Exp,Spec>& sum )
        {
            return sum.a + get<0>( sum.P );
        }
    };
}

template <int idx, typename Scalar, class Exp, class Spec >
__host__ __device__
Scalar get( const ScalarSum<Scalar,Exp,Spec>& sum )
{
    return scalarsum_detail::Get<idx,Scalar,Exp,Spec>::do_it(sum);
}


template <typename Scalar, class Exp, class Spec>
__host__ __device__ __forceinline__
ScalarSum<Scalar,Exp,Spec> operator+( const Scalar& a,
        RValue<Scalar,Exp,Spec> const& B )
{
    return ScalarSum<Scalar,Exp,Spec>(a,
            static_cast<Exp const&>(B));
}

template <typename Scalar, class Exp, class Spec>
__host__ __device__ __forceinline__
ScalarSum<Scalar,Exp,Spec> operator+(
        RValue<Scalar,Exp,Spec> const& B,
        const Scalar& a)
{
    return ScalarSum<Scalar,Exp,Spec>(a,
            static_cast<Exp const&>(B));
}




} // polynomial
} // cuda
} // mpblocks





#endif // SUM_H_
