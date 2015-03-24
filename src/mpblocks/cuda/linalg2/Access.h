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
 *  @file   mpblocks/cuda/linalg2/Access.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_ACCESS_H_
#define MPBLOCKS_CUDA_LINALG2_ACCESS_H_


namespace mpblocks {
namespace cuda     {
namespace linalg2  {




template < Size_t i, Size_t j,
            typename Scalar, Size_t ROWS, Size_t COLS,
            class    Exp >
__host__ __device__
Scalar get( const RValue<Scalar,ROWS,COLS,Exp>& M )
{
    return static_cast<Exp const&>(M).Exp::template me<i,j>();
}

template < Size_t i,
            typename Scalar, Size_t ROWS, Size_t COLS,
            class    Exp >
__host__ __device__
Scalar get( const RValue<Scalar,ROWS,COLS,Exp>& M )
{
    return static_cast<Exp const&>(M).Exp::template ve<i>();
}

template < Size_t i, Size_t j,
            typename Scalar, Size_t ROWS, Size_t COLS,
            class    Exp >
__host__ __device__
Scalar& set( LValue<Scalar,ROWS,COLS,Exp>& M )
{
    return static_cast<Exp&>(M).Exp::template me<i,j>();
}

template < Size_t i,
            typename Scalar, Size_t ROWS, Size_t COLS,
            class    Exp >
__host__ __device__
Scalar& set( LValue<Scalar,ROWS,COLS,Exp>& M )
{
    return static_cast<Exp&>(M).Exp::template ve<i>();
}







} // linalg
} // cuda
} // mpblocks





#endif // LINALG_H_
