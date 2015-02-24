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
 *  @file   pblocks/cuda/linalg/Normalize.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG_NORMALIZE_H_
#define MPBLOCKS_CUDA_LINALG_NORMALIZE_H_

namespace mpblocks {
namespace cuda     {
namespace linalg   {

/// expression template for difference of two matrix expressions
template <typename Scalar, class Exp>
class Normalize:
    public RValue< Scalar, Normalize< Scalar, Exp> >
{
    Scalar     m_norm;
    Exp const& m_A;

    public:
        typedef unsigned int Size_t;

        __device__ __host__
        Normalize( Exp const& A ):
            m_norm(A.norm()),
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
            return m_A[i] / m_norm;
        }

        /// return the evaluated (j,i)'th element of a matrix expression
        __device__ __host__
        Scalar operator()( Size_t i, Size_t j ) const
        {
            return m_A(j,i) / m_norm;
        }
};




template <typename Scalar, class Exp>
__device__ __host__
Normalize< Scalar, Exp > normalize( RValue<Scalar,Exp> const& A )
{
    return Normalize< Scalar,Exp>( static_cast< Exp const& >(A) );
}




} // linalg
} // cuda
} // mpblocks





#endif // NORMALIZE_H_
