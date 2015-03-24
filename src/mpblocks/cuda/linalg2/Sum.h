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
 *  @file   pblocks/cuda/linalg2/Sum.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_SUM_H_
#define MPBLOCKS_CUDA_LINALG2_SUM_H_

namespace mpblocks {
namespace cuda     {
namespace linalg2  {

/// expression template for sum of two expressions
template <typename Scalar, Size_t ROWS, Size_t COLS, class Exp1, class Exp2>
class Sum:
    public RValue< Scalar, ROWS, COLS, Sum< Scalar, ROWS, COLS, Exp1, Exp2> >
{
    Exp1 const& m_A;
    Exp2 const& m_B;

    public:
        __device__ __host__
        Sum( Exp1 const& A, Exp2 const& B ):
            m_A(A),
            m_B(B)
        {}

        /// return the evaluated i'th element of a vector expression
        template< Size_t i >
        __device__ __host__
        Scalar ve() const
        {
            return m_A.Exp1::template ve<i>()
                    + m_B.Exp2::template ve<i>();
        }

        /// return the evaluated (j,i)'th element of a matrix expression
        template< Size_t i, Size_t j >
        __device__ __host__
        Scalar me() const
        {
            return m_A.Exp1::template me<i,j>()
                    + m_B.Exp2::template me<i,j>();
        }
};




template <typename Scalar, Size_t ROWS, Size_t COLS, class Exp1, class Exp2>
__device__ __host__ inline
Sum< Scalar, ROWS, COLS, Exp1, Exp2 >
    operator+( RValue<Scalar,ROWS,COLS,Exp1> const& A,
               RValue<Scalar,ROWS,COLS,Exp2> const& B )
{
    return Sum< Scalar, ROWS, COLS, Exp1,Exp2 >(
                    static_cast< Exp1 const& >(A),
                    static_cast< Exp2 const& >(B) );
}





} // linalg
} // cuda
} // mpblocks





#endif // SUM_H_
