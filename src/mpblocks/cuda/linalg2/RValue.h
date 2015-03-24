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
 *  @file   pblocks/cuda/linalg2/RValue.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_RVALUE_H_
#define MPBLOCKS_CUDA_LINALG2_RVALUE_H_

namespace mpblocks {
namespace cuda     {
namespace linalg2  {

/// expression template for rvalues
template <typename Scalar, Size_t ROWS, Size_t COLS, class Exp>
class RValue
{
    public:
        /// return the evaluated i'th element of a vector expression
        template< Size_t i >
        __device__ __host__
        Scalar ve()
        {
            return static_cast<Exp const&>(*this).Exp::template ve<i>();
        }

        /// return the evaluated (i,j)'th element of a matrix expression
        template< Size_t i, Size_t j >
        __device__ __host__
        Scalar me()
        {
            std::cout << "RValue: returning (" << i << "," << j << ") component\n";
            return static_cast<Exp const&>(*this).Exp::template me<i,j>();
        }

};





} // linalg
} // cuda
} // mpblocks





#endif // MATRIXEXPRESSION_H_
