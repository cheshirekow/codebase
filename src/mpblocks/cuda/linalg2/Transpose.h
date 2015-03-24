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
 *  @file   pblocks/cuda/linalg2/Transpose.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_TRANSPOSE_H_
#define MPBLOCKS_CUDA_LINALG2_TRANSPOSE_H_

namespace mpblocks {
namespace cuda     {
namespace linalg2  {

/// expression template for sum of two expressions
template <typename Scalar, Size_t ROWS, Size_t COLS, class Exp>
class Transpose:
    public RValue< Scalar, COLS, ROWS, Transpose< Scalar, ROWS, COLS, Exp> >
{
    Exp const& m_A;

    public:
        __device__ __host__
        Transpose( Exp const& A ):
            m_A(A)
        {}

        /// return the evaluated i'th element of a vector expression
        template< Size_t i >
        __device__ __host__
        Scalar ve() const
        {
            return m_A.Exp::template ve<i>();
        }

        /// return the evaluated (j,i)'th element of a matrix expression
        template< Size_t i, Size_t j >
        __device__ __host__
        Scalar me() const
        {
            return m_A.Exp::template me<j,i>();
        }
};




template <typename Scalar, Size_t ROWS, Size_t COLS, class Exp>
__device__ __host__ inline
Transpose< Scalar, ROWS, COLS, Exp >
    transpose( RValue<Scalar,ROWS,COLS,Exp> const& A )
{
    return Transpose< Scalar, ROWS, COLS, Exp >(
                        static_cast< Exp const& >(A) );
}





} // linalg
} // cuda
} // mpblocks





#endif // TRANSPOSE_H_
