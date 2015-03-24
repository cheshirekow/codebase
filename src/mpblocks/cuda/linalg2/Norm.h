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
 *  @file   /home/josh/Codes/cpp/mpblocks2/examples/cuda_trajopt/include/mpblocks/cuda/linalg2/Norm.h
 *
 *  @date   Jun 13, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_NORM_H_
#define MPBLOCKS_CUDA_LINALG2_NORM_H_


namespace mpblocks {
namespace cuda     {
namespace linalg2  {


namespace norm_private {


/// returns the square of the (i,j)th element plus the result of the
/// next iterator
template <typename Scalar,
            Size_t ROWS, Size_t COLS,
            Size_t i,    Size_t j,
            class Exp >
struct Iterator
{
    __device__ __host__
    static Scalar result( const Exp& A )
    {
        Scalar r  = A.Exp::template me<i,j>();
               r *= r;
        return r + Iterator<Scalar,ROWS,COLS,i,j+1,Exp>::result(A);
    }
};

/// specialization for past the end iterator intermediate rows
/// returns the result from the next row, 0th column
template <typename Scalar,
            Size_t ROWS, Size_t COLS,
            Size_t i,
            class Exp >
struct Iterator<Scalar,ROWS,COLS,i,COLS,Exp>
{
    __device__ __host__
    static Scalar result( const Exp& A )
    {
        return Iterator<Scalar,ROWS,COLS,i+1,0,Exp>::result(A);
    }
};

/// specialization for past the end iterator for the last row
template <typename Scalar,
            Size_t ROWS, Size_t COLS,
            class Exp >
struct Iterator<Scalar,ROWS,COLS,ROWS,COLS,Exp>
{
    __device__ __host__
    static Scalar result( const Exp& A )
    {
        return 0;
    }
};





/// returns the square of the ith element plus the result of the
/// next iterator
template <typename Scalar,
            Size_t ROWS,
            Size_t i,
            class Exp >
struct Iterator<Scalar,ROWS,1,i,0,Exp>
{
    __device__ __host__
    static Scalar result( const Exp& A )
    {
        Scalar r  = A.Exp::template ve<i>();
               r *= r;
        return r + Iterator<Scalar,ROWS,1,i+1,0,Exp>::result(A);
    }
};

/// specialization for past the end iterator intermediate rows
/// returns the result from the next row, 0th column
template <typename Scalar,
            Size_t ROWS,
            class Exp >
struct Iterator<Scalar,ROWS,1,ROWS,0,Exp>
{
    __device__ __host__
    static Scalar result( const Exp& A )
    {
        return 0;
    }
};






}

/// compute the norm
template <typename Scalar,
            Size_t ROWS, Size_t COLS, class Exp >
__device__ __host__ inline
Scalar norm_squared( const RValue< Scalar, ROWS, COLS, Exp>& M )
{
    return norm_private::Iterator<Scalar,ROWS,COLS,0,0,Exp>::result(
            static_cast<const Exp&>(M) );
};

/// compute the norm
template <typename Scalar,
            Size_t ROWS, Size_t COLS, class Exp >
__device__ __host__ inline
Scalar norm( const RValue< Scalar, ROWS, COLS, Exp>& M )
{
    return std::sqrt( norm_squared(M) );
};


template <typename Scalar, Size_t ROWS, Size_t COLS, class Exp>
__device__ __host__ inline
Scale< Scalar, ROWS, COLS, Exp >
    normalize( RValue<Scalar,ROWS,COLS,Exp> const& A )
{
    return Scale<Scalar,ROWS,COLS,Exp>(
            static_cast<Scalar>(1.0)/norm(A),
            static_cast<Exp const&>(A) );
}







} // linalg
} // cuda
} // mpblocks
















#endif // NORM_H_
