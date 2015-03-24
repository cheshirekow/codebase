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
 *  @file   pblocks/cuda/linalg2/View.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_VIEW_H_
#define MPBLOCKS_CUDA_LINALG2_VIEW_H_

namespace mpblocks {
namespace cuda     {
namespace linalg2  {

/// expression template for subset of a matrix expression
template < Size_t i, Size_t j, Size_t ROWS, Size_t COLS, typename Scalar, class Exp>
class View:
    public RValue< Scalar, ROWS, COLS, View<i,j,ROWS,COLS,Scalar,Exp> >
{
    public:
        typedef unsigned int Size_t;

    protected:
        Exp const& m_A;

    public:
        __device__ __host__
        View( Exp const& A ):
            m_A(A)
        {}

        /// return the evaluated i'th element of a vector expression
        template< Size_t i2 >
        __device__ __host__
        Scalar ve() const
        {
            return m_A.Exp::template ve<i+i2>();
        }

        /// return the evaluated (j,i)'th element of a matrix expression
        template< Size_t i2, Size_t j2 >
        __device__ __host__
        Scalar me() const
        {
            return m_A.Exp::template me<i2+i,j2+j>();
        }
};


/// return an RValue which is a block view of a matrix
template < Size_t i, Size_t j,
            Size_t ROWS, Size_t COLS,
            typename Scalar,
            Size_t ROWS2, Size_t COLS2,
            class Exp>
__device__ __host__
View<i,j,ROWS,COLS,Scalar,Exp> view(
        const RValue< Scalar, ROWS2, COLS2, Exp >& A )
{
    return View<i,j,ROWS,COLS,Scalar,Exp>(
            static_cast<Exp const&>(A) );
};

/// return an RValue which is a subset of a vector
template < Size_t i,
            Size_t ROWS,
            typename Scalar,
            Size_t ROWS2,
            class Exp>
__device__ __host__ inline
View<i,0,ROWS,1,Scalar,Exp> view(
        const RValue< Scalar, ROWS2, 1, Exp >& A )
{
    return View<i,0,ROWS,1,Scalar,Exp>(
            static_cast<Exp const&>(A) );
};




} // linalg
} // cuda
} // mpblocks





#endif // VIEW_H_
