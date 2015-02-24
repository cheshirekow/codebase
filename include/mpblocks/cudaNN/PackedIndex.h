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
 *  @file   /home/josh/Codes/cpp/mpblocks2/dubins/include/mpblocks/dubins/curves_cuda/PackedIndex.h
 *
 *  @date   Jun 20, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_PACKEDINDEX_H_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_PACKEDINDEX_H_

#include <stdint.h>

namespace      mpblocks {
namespace        dubins {
namespace  curves_cuda {





template <unsigned int BYTES>
struct PackedStorage{ typedef uint32_t Result; };

template <> struct PackedStorage<1>{ typedef uint8_t  Result; };
template <> struct PackedStorage<2>{ typedef uint16_t Result; };
template <> struct PackedStorage<4>{ typedef uint32_t Result; };
template <> struct PackedStorage<8>{ typedef uint64_t Result; };


/// stores both the index of the entry and the solution id of it's best solver
/// in a bitfield
template <typename Format_t>
class PackedIndex
{
    public:
        typedef typename PackedStorage< sizeof(Format_t) >::Result Storage_t;

    private:
        Storage_t  m_storage;

    public:
        __device__ __host__
        PackedIndex( Format_t bits );

        __device__ __host__
        PackedIndex( SolutionId id = INVALID, Storage_t idx = 0 );

        __device__ __host__
        SolutionId  getId()  const;

        __device__ __host__
        Storage_t   getIdx() const;

        __device__ __host__
        void setId(  SolutionId id  );

        __device__ __host__
        void setIdx( Storage_t  idx );

        __device__ __host__
        Format_t& getPun();

        __device__ __host__
        void setPun( Format_t bits );

        __device__ __host__
        Storage_t getUnsigned();

        __device__ __host__
        void setUnsigned( Storage_t bits );
};




} // curves
} // dubins
} // mpblocks














#endif // PACKEDINDEX_H_
