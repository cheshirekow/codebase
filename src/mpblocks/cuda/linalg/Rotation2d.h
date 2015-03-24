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
 *  @file   src/linalg/Rotation2d.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG_ROTATION2D_H_
#define MPBLOCKS_CUDA_LINALG_ROTATION2D_H_

#include <iostream>
#include <cassert>
#include <cmath>

namespace mpblocks {
namespace cuda     {
namespace linalg   {


template <typename Scalar>
class Rotation2d :
    public RValue< Scalar, Rotation2d<Scalar> >
{
    protected:
        Scalar    m_sa;
        Scalar    m_ca;

    public:
        __device__ __host__
        Rotation2d( Scalar angle=0 ):
            m_sa(std::sin(angle)),
            m_ca(std::cos(angle))
        {}

        __device__ __host__
        int size() const { return 4; }

        __device__ __host__
        int rows() const { return 2; }

        __device__ __host__
        int cols() const { return 2; }

        /// vector accessor
        __device__ __host__
        Scalar operator[](int i) const
        {
            if( i==0 ) return m_ca;
            if( i==1 ) return m_sa;
            if( i==2 ) return -m_sa;
            return 0;
        }

        /// matrix accessor
        __device__ __host__
        Scalar operator()(int i, int j) const
        {
            if( i == j )
                return m_ca;
            else if( i == 0 )
                return -m_sa;
            else
                return m_sa;
        }
};







} // linalg
} // cuda
} // mpblocks





#endif // LINALG_H_
