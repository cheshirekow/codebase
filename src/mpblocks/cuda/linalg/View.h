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
 *  @file   pblocks/cuda/linalg/View.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG_VIEW_H_
#define MPBLOCKS_CUDA_LINALG_VIEW_H_

namespace mpblocks {
namespace cuda     {
namespace linalg   {

/// expression template for subset of a matrix expression
template <typename Scalar, class Exp>
class RView:
    public RValue< Scalar, RView< Scalar, Exp> >
{
    public:
        typedef unsigned int Size_t;

    protected:
        Exp const& m_A;
        Size_t  m_i;
        Size_t  m_j;
        Size_t  m_rows;
        Size_t  m_cols;

    public:

        __device__ __host__
        RView( Exp const& A, Size_t i, Size_t j, Size_t rows, Size_t cols ):
            m_A(A),
            m_i(i),
            m_j(j),
            m_rows(rows),
            m_cols(cols)
        {}

        /// return the size for a vector
        __device__ __host__
        Size_t size() const
        {
            return m_rows * m_cols;
        }

        /// return the rows of a matrix expression
        __device__ __host__
        Size_t rows() const
        {
            return m_rows;
        }

        /// return the columns of a matrix expression
        __device__ __host__
        Size_t cols() const
        {
            return m_cols;
        }

        /// return the evaluated i'th element of a vector expression
        __device__ __host__
        Scalar operator[]( Size_t k ) const
        {
            Size_t i = k/m_cols;
            Size_t j = k - i*m_cols;
            return (*this)(i,j);
        }

        /// return the evaluated (j,i)'th element of a matrix expression
        __device__ __host__
        Scalar operator()( Size_t i, Size_t j ) const
        {
            return m_A(m_i+i,m_j+j);
        }
};




template <typename Scalar, class Exp>
__device__ __host__
RView< Scalar, Exp > view( RValue<Scalar,Exp> const& A,
                            int i,    int j,
                            int rows, int cols)
{
    return RView< Scalar, Exp >(
                static_cast< Exp const& >(A),i,j,rows,cols );
}

template <typename Scalar, class Exp>
__device__ __host__
RView< Scalar, Exp > rowView( RValue<Scalar,Exp> const& A, int i)
{
    return RView< Scalar,Exp >(
                static_cast< Exp const& >(A),i,0,1,A.cols() );
}

template <typename Scalar, class Exp>
__device__ __host__
RView< Scalar, Exp > columnView( RValue<Scalar,Exp> const& A, int j)
{
    return RView< Scalar,Exp >(
                static_cast< Exp const& >(A),0,j,A.rows(),1 );
}








/// expression template for subset of a matrix expression
template <typename Scalar, class Exp>
class LView:
    public LValue< Scalar, LView< Scalar, Exp> >,
    public RValue< Scalar, LView< Scalar, Exp> >
{
    public:
        typedef unsigned int Size_t;

    protected:
        Exp &   m_A;
        Size_t  m_i;
        Size_t  m_j;
        Size_t  m_rows;
        Size_t  m_cols;

    public:

        __device__ __host__
        LView( Exp& A, Size_t i, Size_t j, Size_t rows, Size_t cols ):
            m_A(A),
            m_i(i),
            m_j(j),
            m_rows(rows),
            m_cols(cols)
        {}

        /// return the size for a vector
        __device__ __host__
        Size_t size() const
        {
            return m_rows * m_cols;
        }

        /// return the rows of a matrix expression
        __device__ __host__
        Size_t rows() const
        {
            return m_rows;
        }

        /// return the columns of a matrix expression
        __device__ __host__
        Size_t cols() const
        {
            return m_cols;
        }

        /// return the evaluated i'th element of a vector expression
        __device__ __host__
        Scalar const& operator[]( Size_t k ) const
        {
            Size_t i = k/m_cols;
            Size_t j = k - i*m_cols;
            return (*this)(i,j);
        }

        /// return the evaluated i'th element of a vector expression
        __device__ __host__
        Scalar& operator[]( Size_t k )
        {
            Size_t i = k/m_cols;
            Size_t j = k - i*m_cols;
            return (*this)(i,j);
        }

        /// return the evaluated (j,i)'th element of a matrix expression
        __device__ __host__
        Scalar const& operator()( Size_t i, Size_t j ) const
        {
            return m_A(m_i+i,m_j+j);
        }

        /// return the evaluated (j,i)'th element of a matrix expression
        __device__ __host__
        Scalar& operator()( Size_t i, Size_t j )
        {
            return m_A(m_i+i,m_j+j);
        }

        template <class Exp2>
        __device__ __host__
        LView<Scalar,Exp>& operator=( RValue<Scalar,Exp2> const& B )
        {
            LValue<Scalar, LView<Scalar,Exp> >::operator=(B);
            return *this;
        }

        template <class Exp2>
        __device__ __host__
        LView<Scalar,Exp>& operator=( LValue<Scalar,Exp2> const& B )
        {
            LValue<Scalar, LView<Scalar,Exp> >::operator=(B);
            return *this;
        }
};




template <typename Scalar, class Exp>
__device__ __host__
LView< Scalar, Exp > block( LValue<Scalar,Exp>& A,
                            int i,    int j,
                            int rows, int cols)
{
    return LView< Scalar, Exp >(
                static_cast< Exp& >(A),i,j,rows,cols );
}

template <typename Scalar, class Exp>
__device__ __host__
LView< Scalar, Exp > row( LValue<Scalar,Exp>& A, int i)
{
    return LView< Scalar,Exp >(
                static_cast< Exp& >(A),i,0,1,A.cols() );
}

template <typename Scalar, class Exp>
__device__ __host__
LView< Scalar, Exp > column( LValue<Scalar,Exp>& A, int j)
{
    return LView< Scalar,Exp >(
                static_cast< Exp& >(A),0,j,A.rows(),1 );
}




} // linalg
} // cuda
} // mpblocks





#endif // VIEW_H_
