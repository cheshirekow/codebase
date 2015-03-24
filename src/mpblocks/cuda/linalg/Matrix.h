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
 *  @file   src/linalg/Matrix.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG_MATRIX_H_
#define MPBLOCKS_CUDA_LINALG_MATRIX_H_

#include <iostream>
#include <cassert>

namespace mpblocks {
namespace cuda     {
namespace linalg   {


template <typename Scalar, int ROWS, int COLS>
class Matrix :
    public LValue< Scalar, Matrix<Scalar,ROWS,COLS> >,
    public RValue< Scalar, Matrix<Scalar,ROWS,COLS> >
{
    public:
        typedef Matrix<Scalar,ROWS,COLS >           Matrix_t;
        typedef LValue< Scalar, Matrix_t >          LValue_t;


    protected:
        Scalar    m_data[ROWS*COLS];

    public:
        __device__ __host__
        int size() const { return ROWS*COLS; }

        __device__ __host__
        int rows() const { return ROWS; }

        __device__ __host__
        int cols() const { return COLS; }

        /// vector accessor
        __device__ __host__
        Scalar&   operator[](int i)
        {
            return m_data[i];
        }

        /// vector accessor
        __device__ __host__
        Scalar const&   operator[](int i) const
        {
            return m_data[i];
        }

        /// matrix accessor
        __device__ __host__
        Scalar&   operator()(int i, int j)
        {
//            assert(i<ROWS && j<COLS);
            return m_data[i*COLS +j];
        }

        /// matrix accessor
        __device__ __host__
        Scalar const&   operator()(int i, int j) const
        {
//            assert(i<ROWS && j<COLS);
            return m_data[i*COLS +j];
        }

        /// Default constructor
        __device__ __host__
        Matrix(){}


        /// Construct from any MatrixExpression:
        template <typename Exp>
        __device__ __host__
        Matrix( RValue<Scalar,Exp> const& exp )
        {
//            assert( exp.rows() == rows() );
//            assert( exp.cols() == cols() );

            for( int i=0; i < rows(); i++)
            {
                for(int j=0; j < cols(); j++)
                {
                    (*this)(i,j) = exp(i,j);
                }
            }
        }
};







} // linalg
} // cuda
} // mpblocks





#endif // LINALG_H_
