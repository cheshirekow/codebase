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
 *  @file   mpblocks/cuda/linalg2/AssignmentIterator.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_ASSIGNMENTITERATOR_H_
#define MPBLOCKS_CUDA_LINALG2_ASSIGNMENTITERATOR_H_

#include <iostream>
#include <cassert>

namespace mpblocks {
namespace cuda     {
namespace linalg2  {




/// iterates over elements
template <typename Scalar, Size_t ROWS, Size_t COLS, Size_t i, Size_t j, class Exp>
class AssignmentIterator
{
    private:
        Exp& m_A;

    public:
        __device__ __host__
        AssignmentIterator( Exp& A ):
            m_A(A)
        {}

        __device__ __host__
        AssignmentIterator<Scalar,ROWS,COLS,i,j+1,Exp> operator,( Scalar x )
        {
            m_A.Exp::template me<i,j>() = x;
            return AssignmentIterator<Scalar,ROWS,COLS,i,j+1,Exp>(m_A);
        }
};


/// specialization for past-end-of row, acts like the next row
template <typename Scalar, Size_t ROWS, Size_t COLS, Size_t i, class Exp>
class AssignmentIterator<Scalar,ROWS,COLS,i,COLS,Exp>
{
    private:
        Exp& m_A;

    public:
        __device__ __host__
        AssignmentIterator( Exp& A ):
            m_A(A)
        {}

        __device__ __host__
        AssignmentIterator<Scalar,ROWS,COLS,i+1,1,Exp> operator,( Scalar x)
        {
            m_A.Exp::template me<i+1,0>() = x;
            return AssignmentIterator<Scalar,ROWS,COLS,i+1,1,Exp>(m_A);
        }

};







} // linalg
} // cuda
} // mpblocks





#endif // LINALG_H_
