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
 *  @file   include/mpblocks/polynomial/Difference.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_DERIVATIVE_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_DERIVATIVE_H_

namespace   mpblocks {
namespace       cuda {
namespace polynomial {


namespace derivative_detail
{
    template< class Spec >
    struct Index
    {
        enum { enabled = false };
    };

    template< int Head, class Tail>
    struct Index< IntList<Head,Tail> >
    {
        enum { enabled = (Head > 0) };
    };

    template< bool Enable, class Spec >
    struct DerivativeSpec
    {
        typedef intlist::Terminal result;
    };

    template< int Head, class Tail >
    struct DerivativeSpec< true, IntList<Head,Tail> >
    {
        typedef IntList< Head-1,
            typename DerivativeSpec<
                Index<Tail>::enabled , Tail >::result > result;
    };

}



/// expression template for sum of two matrix expressions
template <typename Scalar, class Exp, class Spec>
struct Derivative:
    public RValue<Scalar, Derivative<Scalar,Exp,Spec>,
        typename derivative_detail::DerivativeSpec<true,Spec>::result >
{
    Exp const& P;

    public:
        __host__ __device__
        Derivative( const Exp& P ):
            P(P)
        {}
};



template <typename Scalar, class Exp, class Spec>
struct get_spec< Derivative<Scalar,Exp,Spec> >
{
    typedef typename
        derivative_detail::DerivativeSpec<true,Spec>::result result;
};


template <int idx, typename Scalar, class Exp, class Spec>
__host__ __device__
Scalar get( const Derivative<Scalar,Exp,Spec>& exp )
{
    return (idx+1)*get<idx+1>( exp.P );
}




template <typename Scalar, class Exp, class Spec>
__host__ __device__
Derivative<Scalar,Exp,Spec> d_dx( RValue<Scalar,Exp,Spec> const& P )
{
    return Derivative<Scalar,Exp,Spec>( static_cast<Exp const&>(P) );
}




} // polynomial
} // cuda
} // mpblocks



#endif // DERIVATIVE_H_
