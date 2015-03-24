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
 *  @file   /home/josh/Codes/cpp/mpblocks2/examples/cuda_trajopt/include/mpblocks/cuda/linalg2/mktmp.h
 *
 *  @date   Jun 14, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_MKTMP_H_
#define MPBLOCKS_CUDA_LINALG2_MKTMP_H_



namespace mpblocks {
namespace cuda     {
namespace linalg2  {



/// forces the creation of a temporary
template <typename Scalar, Size_t ROWS, Size_t COLS, class Exp >
__device__ __host__ inline
Matrix< Scalar, ROWS, COLS  >
    mktmp( RValue<Scalar,ROWS,COLS,Exp> const& M )
{
    return Matrix< Scalar, ROWS, COLS >(M);
}





} // linalg
} // cuda
} // mpblocks




#endif // MKTMP_H_
