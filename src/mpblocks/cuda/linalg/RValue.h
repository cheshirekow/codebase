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
 *  @file   pblocks/cuda/linalg/RValue.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG_RVALUE_H_
#define MPBLOCKS_CUDA_LINALG_RVALUE_H_

#include <cmath>

namespace mpblocks {
namespace cuda     {
namespace linalg   {

/// forward declared for normalized()
template <typename Scalar, class Exp> class Scale;

/// expression template for rvalues
template <typename Scalar, class Mat>
class RValue
{
    public:
        typedef unsigned int Size_t;

        /// return the size for a vector
        __device__ __host__
        Size_t size() const
        {
            return static_cast<Mat const&>(*this).size();
        }

        /// return the rows of a matrix expression
        __device__ __host__
        Size_t rows() const
        {
            return static_cast<Mat const&>(*this).rows();
        }

        /// return the columns of a matrix expression
        __device__ __host__
        Size_t cols() const
        {
            return static_cast<Mat const&>(*this).cols();
        }

        /// return the evaluated i'th element of a vector expression
        __device__ __host__
        Scalar operator[]( Size_t i ) const
        {
            return static_cast<Mat const&>(*this)[i];
        }

        /// return the evaluated (i,j)'th element of a matrix expression
        __device__ __host__
        Scalar operator()( Size_t i, Size_t j ) const
        {
            return static_cast<Mat const&>(*this)(i,j);
        }

        __device__ __host__
        Scalar norm_squared() const
        {
            Scalar r=0;
            for(int i=0; i < size(); i++)
                r += (*this)[i] * (*this)[i];
            return r;
        }

        __device__ __host__
        Scalar norm() const
        {
            return std::sqrt( norm_squared() );
        }

        __device__ __host__
        Scale< Scalar, Mat > normalized() const
        {
            Scalar scale = 1.0/norm();
            return Scale<Scalar,Mat>( scale, static_cast<Mat const&>(*this) );
        }

        __device__ __host__
        Scalar maxCoeff()
        {
            Scalar r = (*this)[0];
            for(int i=1; i < size(); i++)
                if( (*this)[i] > r )
                    r = (*this)[i];
            return r;
        }


};




} // linalg
} // cuda
} // mpblocks





#endif // MATRIXEXPRESSION_H_
