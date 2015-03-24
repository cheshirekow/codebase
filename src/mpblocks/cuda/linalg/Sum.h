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
 *  @file   pblocks/cuda/linalg/Sum.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG_SUM_H_
#define MPBLOCKS_CUDA_LINALG_SUM_H_

namespace mpblocks {
namespace cuda     {
namespace linalg   {

/// expression template for sum of two matrix expressions
template <typename Scalar, class Exp1, class Exp2>
class Sum :
    public RValue<Scalar, Sum<Scalar,Exp1,Exp2> >
{
    Exp1 const& m_A;
    Exp2 const& m_B;

    public:
        typedef unsigned int Size_t;

        __device__ __host__
        Sum( Exp1 const& A, Exp2 const& B ):
            m_A(A),
            m_B(B)
        {
//            assert( A.size() == B.size() );
//            assert( A.rows() == B.rows() );
//            assert( A.cols() == B.cols() );
        }

        /// return the size for a vector
        __device__ __host__
        Size_t size() const
        {
            return m_A.size();
        }

        /// return the rows of a matrix expression
        __device__ __host__
        Size_t rows() const
        {
            return m_A.rows();
        }

        /// return the columns of a matrix expression
        __device__ __host__
        Size_t cols() const
        {
            return m_A.cols();
        }

        /// return the evaluated i'th element of a vector expression
        __device__ __host__
        Scalar operator[]( Size_t i ) const
        {
            return (m_A[i] + m_B[i]);
        }

        /// return the evaluated (i,j)'th element of a matrix expression
        __device__ __host__
        Scalar operator()( Size_t i, Size_t j )const
        {
            return ( m_A(i,j) + m_B(i,j) );
        }
};




template <typename Scalar, class Exp1, class Exp2>
__device__ __host__
Sum<Scalar,Exp1,Exp2> operator+(
        RValue<Scalar,Exp1> const& A, RValue<Scalar,Exp2> const& B )
{
    typedef Sum<Scalar,Exp1,Exp2> Sum_t;
    return Sum_t(
            static_cast<Exp1 const&>(A),
            static_cast<Exp2 const&>(B));
}




} // linalg
} // cuda
} // mpblocks





#endif // SUM_H_
