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
 *  @file   mpblocks/cuda/linalg2/Assignment.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_ASSIGNMENT_H_
#define MPBLOCKS_CUDA_LINALG2_ASSIGNMENT_H_

#include <iostream>
#include <cassert>

namespace mpblocks {
namespace cuda     {
namespace linalg2  {




/// copies elements one by one
template <typename Scalar,
                Size_t ROWS, Size_t COLS,
                Size_t i,    Size_t j,
                class Exp1,  class Exp2>
class Assignment:
    public Assignment< Scalar, ROWS,COLS,i,j+1,Exp1,Exp2>
{
    public:
        __device__ __host__
        Assignment( Exp1& A, Exp2 const& B ):
            Assignment< Scalar, ROWS,COLS,i,j+1,Exp1,Exp2>(A,B)
        {
            A.Exp1::template me<i,j>() = B.Exp2::template me<i,j>();
        }
};


/// specialization for past-end-of row, acts like the next row
template <typename Scalar,
                Size_t ROWS, Size_t COLS,
                Size_t i,
                class Exp1,  class Exp2>
class Assignment<Scalar,ROWS,COLS,i,COLS,Exp1,Exp2>:
    public Assignment<Scalar,ROWS,COLS,i+1,0,Exp1,Exp2>
{
    public:
        __device__ __host__
        Assignment( Exp1& A, Exp2 const& B ):
            Assignment<Scalar,ROWS,COLS,i+1,0,Exp1,Exp2>(A,B)
        {

        }
};

/// specialization for past-end-of array
template <typename Scalar,
                Size_t ROWS, Size_t COLS,
                class Exp1,  class Exp2>
class Assignment<Scalar,ROWS,COLS,ROWS,0,Exp1,Exp2>
{
    public:
        __device__ __host__
        Assignment( Exp1& A, Exp2 const& B ){}
};






template <typename Scalar,
                Size_t ROWS, Size_t COLS,
                Size_t i,    Size_t j,
                class Exp1,  class Exp2>
class Assignment2
{
    public:
        __device__ __host__
        static void doit( Exp1& A, Exp2 const& B )
        {
            A.Exp1::template me<i,j>() = B.Exp2::template me<i,j>();
            Assignment2< Scalar, ROWS,COLS,i,j+1,Exp1,Exp2>::doit(A,B);
        }
};


/// specialization for past-end-of row, acts like the next row
template <typename Scalar,
                Size_t ROWS, Size_t COLS,
                Size_t i,
                class Exp1,  class Exp2>
class Assignment2<Scalar,ROWS,COLS,i,COLS,Exp1,Exp2>
{
    public:
        __device__ __host__
        static void doit( Exp1& A, Exp2 const& B )
        {
            Assignment2<Scalar,ROWS,COLS,i+1,0,Exp1,Exp2>::doit(A,B);
        }
};

/// specialization for past-end-of array
template <typename Scalar,
                Size_t ROWS, Size_t COLS,
                class Exp1,  class Exp2>
class Assignment2<Scalar,ROWS,COLS,ROWS,0,Exp1,Exp2>
{
    public:
        __device__ __host__
        static void doit( Exp1& A, Exp2 const& B ){}
};







} // linalg
} // cuda
} // mpblocks





#endif // LINALG_H_
