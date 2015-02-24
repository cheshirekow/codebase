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
 *  @file   pblocks/cuda/linalg/Scale.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG_SCALE_H_
#define MPBLOCKS_CUDA_LINALG_SCALE_H_

namespace mpblocks {
namespace cuda     {
namespace linalg   {

/// expression template for multiplication by a scalar
template <typename Scalar, class Exp>
class Scale:
    public RValue< Scalar, Scale< Scalar, Exp> >
{
    Scalar     m_s;
    Exp const& m_A;

    public:
        typedef unsigned int Size_t;

        __device__ __host__
        Scale( Scalar s, Exp const& A ):
            m_s(s),
            m_A(A)
        {}

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
            return m_A.cols();
        }

        /// return the columns of a matrix expression
        __device__ __host__
        Size_t cols() const
        {
            return m_A.rows();
        }

        /// return the evaluated i'th element of a vector expression
        __device__ __host__
        Scalar operator[]( Size_t i ) const
        {
            return m_s * m_A[i];
        }

        /// return the evaluated (j,i)'th element of a matrix expression
        __device__ __host__
        Scalar operator()( Size_t i, Size_t j ) const
        {
            return m_s * m_A(i,j);
        }
};




template <typename Scalar, class Exp>
__device__ __host__
Scale< Scalar, Exp > operator*( Scalar s, RValue<Scalar,Exp> const& A )
{
    return Scale< Scalar,Exp>( s, static_cast< Exp const& >(A) );
}

template <typename Scalar, class Exp>
__device__ __host__
Scale< Scalar, Exp > operator*( RValue<Scalar,Exp> const& A, Scalar s )
{
    return Scale< Scalar,Exp>( s, static_cast< Exp const& >(A) );
}




} // linalg
} // cuda
} // mpblocks





#endif // SCALE_H_
