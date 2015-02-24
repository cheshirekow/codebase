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
 *  @file   include/mpblocks/polynomial/Negative.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_NEGATIVE_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_NEGATIVE_H_

namespace   mpblocks {
namespace       cuda {
namespace polynomial {

/// expression template for sum of two matrix expressions
template <typename Scalar,class Exp,class Spec>
struct Negative :
    public RValue< Scalar,Negative<Scalar,Exp,Spec>,Spec >
{
    Exp const& exp;

    __host__ __device__
    Negative( Exp const& exp ):
        exp(exp)
    {}

    __host__ __device__
    Scalar eval( Scalar x )
    {
        return -exp.eval(x);
    }
};

template <typename Scalar, class Exp, class Spec>
struct get_spec< Negative<Scalar,Exp,Spec > >
{
    typedef Spec result;
};

template <int idx, typename Scalar, class Exp, class Spec>
__host__ __device__
Scalar get( const Negative<Scalar,Exp,Spec>& exp )
{
    return -get<idx>( exp.exp );
}


template <typename Scalar, class Exp, class Spec>
__host__ __device__
Negative<Scalar,Exp,Spec> operator-(
        RValue<Scalar,Exp,Spec> const& exp )
{
    return Negative<Scalar,Exp,Spec>( static_cast<Exp const&>(exp) );
}




} // polynomial
} // cuda
} // mpblocks





#endif // NEGATIVE_H_
