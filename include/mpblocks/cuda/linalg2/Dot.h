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
 *  @file   /home/josh/Codes/cpp/mpblocks2/examples/cuda_trajopt/include/mpblocks/cuda/linalg2/DOT.h
 *
 *  @date   Jun 13, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_DOT_H_
#define MPBLOCKS_CUDA_LINALG2_DOT_H_


namespace mpblocks {
namespace cuda     {
namespace linalg2  {


namespace dot_private {




/// returns the square of the ith element plus the result of the
/// next iterator
template <typename Scalar,
            Size_t ROWS,
            Size_t i,
            class ExpA,
            class ExpB>
struct Iterator
{
    __device__ __host__
    static Scalar result( const ExpA& A, const ExpB& B )
    {
        return A.ExpA::template ve<i>() * B.ExpB::template ve<i>()
                + Iterator<Scalar,ROWS,i+1,ExpA,ExpB>::result(A,B);
    }
};

/// specialization for past the end iterator intermediate rows
/// returns the result from the next row, 0th column
template <typename Scalar,
            Size_t ROWS,
            class ExpA,
            class ExpB >
struct Iterator<Scalar,ROWS,ROWS,ExpA,ExpB>
{
    __device__ __host__
    static Scalar result( const ExpA& A, const ExpB& B )
    {
        return 0;
    }
};






}

/// compute the DOT
template <typename Scalar,
            Size_t ROWS, class ExpA, class ExpB >
__device__ __host__ inline
Scalar dot( const RValue< Scalar, ROWS, 1, ExpA>& A,
            const RValue< Scalar, ROWS, 1, ExpB>& B)
{
    return dot_private::Iterator<Scalar,ROWS,0,ExpA,ExpB>::result(
            static_cast<const ExpA&>(A),
            static_cast<const ExpB&>(B) );
};





} // linalg
} // cuda
} // mpblocks
















#endif // DOT_H_
