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
 *  @file   pblocks/cuda/linalg2/RValue.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_LVALUE_H_
#define MPBLOCKS_CUDA_LINALG2_LVALUE_H_

namespace mpblocks {
namespace cuda     {
namespace linalg2  {

/// expression template for rvalues
template <typename Scalar, Size_t ROWS, Size_t COLS, class Exp>
class LValue
{
    public:
        /// return a mutable reference to the i'th element of the vector
        /// expression
        template< Size_t i >
        __device__ __host__
        Scalar& ve()
        {
            return static_cast<Exp&>(*this).Exp::template ve<i>();
        }

        /// return an immutable reference to the i'th element of the vector
        /// expression
        template< Size_t i >
        __device__ __host__
        Scalar const& ve() const
        {
            return static_cast<Exp const&>(*this).Exp::template ve<i>();
        }

        /// return a mutable reference to the (i,j)th element of the matrix
        /// expression
        template< Size_t i, Size_t j >
        __device__ __host__
        Scalar& me()
        {
            return static_cast<Exp&>(*this).Exp::template me<i,j>();
        }

        /// return an immutable reference to the (i,j)th element of the matrix
        /// expression
        template< Size_t i, Size_t j >
        __device__ __host__
        Scalar const& me() const
        {
            return static_cast<Exp const&>(*this).Exp::template me<i,j>();
        }

        /// assignment
        template< typename Exp2 >
        __device__ __host__
        LValue<Scalar,ROWS,COLS,Exp>& operator=(
                RValue<Scalar,ROWS,COLS,Exp2> const& B )
        {
            Assignment2<Scalar,ROWS,COLS,0,0,Exp,Exp2>::doit(
                    static_cast<Exp&>(*this),
                    static_cast<Exp2 const&>(B) );

            return *this;
        }

        /// return an assignment iterator
        __device__ __host__
        AssignmentIterator<Scalar,ROWS,COLS,0,1,Exp>
            operator<<( Scalar x )
        {
            me<0,0>() = x;
            return AssignmentIterator<Scalar,ROWS,COLS,0,1,Exp>(
                    static_cast<Exp&>(*this) );
        }
};





} // linalg
} // cuda
} // mpblocks





#endif // MATRIXEXPRESSION_H_
