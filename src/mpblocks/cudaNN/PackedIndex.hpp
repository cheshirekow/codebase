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

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_PACKEDINDEX_HPP_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_PACKEDINDEX_HPP_


namespace      mpblocks {
namespace        dubins {
namespace  curves_cuda {


template <typename Format_t>
__device__ __host__
PackedIndex<Format_t>::PackedIndex( SolutionId id, Storage_t idx )
{
    setId(id);
    setIdx(idx);
}

template <typename Format_t>
__device__ __host__
PackedIndex<Format_t>::PackedIndex( Format_t bits )
{
    setPun(bits);
}

template <typename Format_t>
__device__ __host__
SolutionId  PackedIndex<Format_t>::getId()  const
{
    // the number of bits in Storage_t is sizeof(Format_t)*sizeof(char)
    const Storage_t   size  = sizeof(Format_t)*8;

    // we'll use the first four bits to store the id so
    const Storage_t   field = 4;

    return (SolutionId)( m_storage >> (size-field) );
}

template <typename Format_t>
__device__ __host__
typename PackedIndex<Format_t>::Storage_t PackedIndex<Format_t>::getIdx() const
{
    // the number of bits in Storage_t is sizeof(Format_t)*sizeof(char)
    const Storage_t   size  = sizeof(Format_t)*8;

    // we'll use the first four bits to store the id so
    const Storage_t   field = 4;
    const Storage_t   mask  = ~( 0x0F << (size-field) );

    return (m_storage & mask);
}

template <typename Format_t>
__device__ __host__
void PackedIndex<Format_t>::setId(  SolutionId id  )
{
    const Storage_t   mask  = 0x0FFFFFFF;
    Storage_t idBits = id;
    m_storage = (idBits << (sizeof(Format_t)*8-4) ) | (m_storage & mask);
}

template <typename Format_t>
__device__ __host__
void PackedIndex<Format_t>::setIdx( Storage_t  idx )
{
    const Storage_t idMask  = 0xF0000000;
    const Storage_t idxMask = 0x0FFFFFFF;

    m_storage = ( idMask & m_storage ) | ( idxMask & idx );
}

template <typename Format_t>
__device__ __host__
Format_t& PackedIndex<Format_t>::getPun()
{
    return reinterpret_cast<Format_t&>(m_storage);
}

template <typename Format_t>
__device__ __host__
void PackedIndex<Format_t>::setPun( Format_t bits )
{
    reinterpret_cast<Format_t&>(m_storage) = bits;
}

template <typename Format_t>
__device__ __host__
typename PackedIndex<Format_t>::Storage_t PackedIndex<Format_t>::getUnsigned()
{
    return m_storage;
}

template <typename Format_t>
__device__ __host__
void PackedIndex<Format_t>::setUnsigned( Storage_t bits )
{
    m_storage = bits;
}





} // curves
} // dubins
} // mpblocks














#endif // PACKEDINDEX_H_
