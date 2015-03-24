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
 *  @file   include/mpblocks/polynomial/StreamAssignment.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_STREAMASSIGNMENT_H_
#define MPBLOCKS_CUDA_POLYMOMIAL_STREAMASSIGNMENT_H_

#include <iostream>
#include <cassert>
#include <mpblocks/cuda/polynomial/LValue.h>

namespace   mpblocks {
namespace       cuda {
namespace polynomial {



template< int idx, typename Scalar, class Exp1 >
__host__ __device__
Scalar& set_storage( Exp1& );


/// assignment
template <class Scalar, class Exp, int idx>
class StreamAssignment
{
    private:
        Exp& m_exp;

    public:
        __host__ __device__
        StreamAssignment( Exp& exp ):
            m_exp(exp)
        {}

        template <typename Scalar2>
        __host__ __device__
        StreamAssignment<Scalar,Exp,idx+1> append( Scalar2 x )
        {
            set_storage<idx>(m_exp) = x;
            return StreamAssignment<Scalar,Exp,idx+1>(m_exp);
        }

        template <typename Scalar2>
        __host__ __device__
        StreamAssignment<Scalar,Exp,idx+1> operator,( Scalar2 x )
        {
            return append<Scalar>(x);
        }
};

template <typename Scalar, class Exp, typename Scalar2>
__host__ __device__
StreamAssignment<Scalar,Exp,1> operator<<( LValue<Scalar,Exp>& exp, const Scalar2& val )
{
    set_storage<0>( static_cast<Exp&>(exp) ) = val;
    return StreamAssignment<Scalar,Exp,1>( static_cast<Exp&>(exp) );
}

} // polynomial
} // cuda
} // mpblocks





#endif // POLYMOMIAL_H_
