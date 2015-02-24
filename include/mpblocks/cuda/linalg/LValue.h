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
 *  @file   pblocks/cuda/linalg/LValue.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG_LVALUE_H_
#define MPBLOCKS_CUDA_LINALG_LVALUE_H_

namespace mpblocks {
namespace cuda     {
namespace linalg   {

/// expression template for rvalues
template <typename Scalar, class Mat>
class LValue
{
    public:
        typedef StreamAssignment< LValue<Scalar,Mat> >        Stream_t;

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
        Scalar& operator[]( Size_t i )
        {
            return static_cast<Mat&>(*this)[i];
        }

        /// return the evaluated i'th element of a vector expression
        __device__ __host__
        Scalar const& operator[]( Size_t i )const
        {
            return static_cast<Mat const&>(*this)[i];
        }

        /// return the evaluated (i,j)'th element of a matrix expression
        __device__ __host__
        Scalar& operator()( Size_t i, Size_t j )
        {
            return static_cast<Mat&>(*this)(i,j);
        }

        /// return the evaluated (i,j)'th element of a matrix expression
        __device__ __host__
        Scalar const& operator()( Size_t i, Size_t j )const
        {
            return static_cast<Mat const&>(*this)(i,j);
        }

        /// returns a stream for assignment
        __device__ __host__
        Stream_t operator<<( Scalar x )
        {
            Stream_t stream(*this);
            return   stream << x;
        }

        template <class Exp2>
        __device__ __host__
        LValue<Scalar,Mat>& operator=( RValue<Scalar,Exp2> const& B )
        {
//            assert( rows() == B.rows() );
//            assert( cols() == B.cols() );

            for( int i=0; i < rows(); i++)
                for(int j=0; j < cols(); j++)
                    (*this)(i,j) = B(i,j);
            return *this;
        }

        template <class Exp2>
        __device__ __host__
        LValue<Scalar,Mat>& operator+=( RValue<Scalar,Exp2> const& B )
        {
//            assert( rows() == B.rows() );
//            assert( cols() == B.cols() );

            for( int i=0; i < rows(); i++)
                for(int j=0; j < cols(); j++)
                    (*this)(i,j) += B(i,j);
            return *this;
        }

        template <class Exp2>
        __device__ __host__
        LValue<Scalar,Mat>& operator-=( RValue<Scalar,Exp2> const& B )
        {
//            assert( rows() == B.rows() );
//            assert( cols() == B.cols() );

            for( int i=0; i < rows(); i++)
                for(int j=0; j < cols(); j++)
                    (*this)(i,j) -= B(i,j);
            return *this;
        }

        __device__ __host__
        operator RValue<Scalar,LValue<Scalar,Mat> >()
        {
            return RValue<Scalar, LValue<Scalar,Mat> >(*this);
        }

};






} // linalg
} // cuda
} // mpblocks





#endif // MATRIXEXPRESSION_H_
