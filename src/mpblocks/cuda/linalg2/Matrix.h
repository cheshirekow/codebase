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
 *  @file   src/linalg2/Matrix.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_MATRIX_H_
#define MPBLOCKS_CUDA_LINALG2_MATRIX_H_

#include <iostream>
#include <cassert>

namespace mpblocks {
namespace cuda     {
namespace linalg2  {


/// default template has column inheritence
template <typename Scalar, Size_t ROWS, Size_t COLS, Size_t i, Size_t j>
struct MatrixElement :
    public MatrixElement<Scalar,ROWS,COLS,i,j+1>
{
    Scalar m_data;
};

/// specialization for 0'th column, also inherits row
template <typename Scalar, Size_t ROWS, Size_t COLS, Size_t i>
struct MatrixElement<Scalar,ROWS,COLS,i,0> :
    public MatrixElement<Scalar,ROWS,COLS,i,1>,
    public MatrixElement<Scalar,ROWS,COLS,i+1,0>
{
    Scalar m_data;
};

/// empty class for last row
template <typename Scalar, Size_t ROWS, Size_t COLS>
struct MatrixElement<Scalar,ROWS,COLS,ROWS,0>
{};

/// empty class for last column
template <typename Scalar, Size_t ROWS, Size_t COLS, Size_t i>
struct MatrixElement<Scalar,ROWS,COLS,i,COLS>
{};


template <typename Scalar, Size_t ROWS, Size_t COLS>
class Matrix :
    protected MatrixElement<Scalar,ROWS,COLS,0,0>,
    public LValue< Scalar,ROWS,COLS, Matrix<Scalar,ROWS,COLS> >,
    public RValue< Scalar,ROWS,COLS, Matrix<Scalar,ROWS,COLS> >
{
    public:
        typedef Matrix<Scalar,ROWS,COLS>            Matrix_t;
        typedef LValue<Scalar,ROWS,COLS, Matrix_t > LValue_t;

    public:
        /// vector accessor
        template <Size_t i>
        __device__ __host__
        Scalar&  sve()
        {
//            if( COLS == 1U )
                return static_cast<
                        MatrixElement<Scalar,ROWS,COLS,i,0>& >(*this).m_data;
//            else
//                return static_cast<
//                        MatrixElement<Scalar,ROWS,COLS,0,i>& >(*this).m_data;
        }

        /// vector accessor
        template <Size_t i>
        __device__ __host__
        Scalar gve() const
        {
//            if( COLS == 1U )
                return static_cast<
                    MatrixElement<Scalar,ROWS,COLS,i,0> const& >(*this).m_data;
//            else
//                return static_cast<
//                    MatrixElement<Scalar,ROWS,COLS,0,i> const& >(*this).m_data;
        }

        /// vector accessor
        template <Size_t i>
        __device__ __host__
        Scalar&  ve()
        {
//            if( COLS == 1U )
                return static_cast<
                        MatrixElement<Scalar,ROWS,COLS,i,0>& >(*this).m_data;
//            else
//                return static_cast<
//                        MatrixElement<Scalar,ROWS,COLS,0,i>& >(*this).m_data;
        }

        /// vector accessor
        template <Size_t i>
        __device__ __host__
        Scalar ve() const
        {
//            if( COLS == 1U )
                return static_cast<
                    MatrixElement<Scalar,ROWS,COLS,i,0> const& >(*this).m_data;
//            else
//                return static_cast<
//                    MatrixElement<Scalar,ROWS,COLS,0,i> const& >(*this).m_data;
        }

        /// matrix accessor
        template <Size_t i, Size_t j>
        __device__ __host__
        Scalar&   me()
        {
            return static_cast<
                    MatrixElement<Scalar,ROWS,COLS,i,j>& >(*this).m_data;
        }

        /// matrix accessor
        template <Size_t i, Size_t j>
        __device__ __host__
        Scalar me() const
        {
            return static_cast<
                    MatrixElement<Scalar,ROWS,COLS,i,j> const& >(*this).m_data;
        }

        /// Default constructor
        __device__ __host__
        Matrix(){}


        /// Construct from any MatrixExpression:
        template <typename Exp>
        __device__ __host__
        Matrix( RValue<Scalar,ROWS,COLS,Exp> const& exp )
        {
            LValue_t::operator=(exp);
        }

        /// dynamic accessor for vector expressions
        /**
         *  @note   this totally depends on the memory layout of the compiler,
         *          which may be different for different compilers
         */
        __device__ __host__
        Scalar& operator[](int i)
        {
            Scalar* buf = (Scalar*)(this);
            return buf[ROWS*COLS -1 - i];
        }

        __device__ __host__
        Scalar const& operator[](int i) const
        {
            Scalar* buf = (Scalar*)(this);
            return buf[ROWS*COLS -1 - i];
        }

        /// dynamic accessor for matrix expressions
        /**
         *  @note   this totally depends on the memory layout of the compiler,
         *          which may be different for different compilers
         */
        __device__ __host__
        Scalar& operator()(int i, int j)
        {
            Scalar* buf         = (Scalar*)(this);
            unsigned int off    = 0;
            if(j == 0 )
                off = i;
            else
                off = ROWS + (COLS-1)*(ROWS-1-i) + (j-1);

            if( off > ROWS*COLS-1 )
                return buf[0];

            return buf[ROWS*COLS-1 - off];
        }

        __device__ __host__
        Scalar const& operator()(int i, int j) const
        {
            Scalar* buf         = (Scalar*)(this);
            unsigned int off    = 0;
            if(j == 0 )
                off = i;
            else
                off = ROWS + (COLS-1)*(ROWS-1-i) + (j-1);

            if( off > ROWS*COLS-1 )
                return buf[0];

            return buf[ROWS*COLS-1 - off];
        }
};








} // linalg
} // cuda
} // mpblocks





#endif // LINALG_H_
