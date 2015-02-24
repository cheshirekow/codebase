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

#include <mpblocks/dubins/curves_cuda/PackedIndex.h>

namespace    mpblocks {
namespace      dubins {
namespace curves_cuda {

template <typename PunType>
__device__ __host__
  PackedIndex<PunType>::PackedIndex(SolutionId id, StorageType idx) {
  setId(id);
  setIdx(idx);
}

template <typename PunType>
__device__ __host__
PackedIndex<PunType>::PackedIndex(PunType bits) {
  setPun(bits);
}

template <typename PunType>
__device__ __host__
SolutionId PackedIndex<PunType>::getId() const {
  // the number of bits in StorageType is sizeof(PunType)*sizeof(char)
  const StorageType size = sizeof(PunType) * 8;

  // we'll use the first four bits to store the id so
  const StorageType field = 4;

  return (SolutionId)(m_storage >> (size - field));
}

template <typename PunType>
__device__ __host__
typename PackedIndex<PunType>::StorageType
PackedIndex<PunType>::getIdx() const {
  // the number of bits in StorageType is sizeof(PunType)*sizeof(char)
  const StorageType size = sizeof(PunType) * 8;

  // we'll use the first four bits to store the id so
  const StorageType field = 4;
  const StorageType mask = ~(0x0F << (size - field));

  return (m_storage & mask);
}

template <typename PunType>
__device__ __host__
void PackedIndex<PunType>::setId(SolutionId id) {
  const StorageType mask = 0x0FFFFFFF;
  StorageType idBits = id;
  m_storage = (idBits << (sizeof(PunType) * 8 - 4)) | (m_storage & mask);
}

template <typename PunType>
__device__ __host__
void PackedIndex<PunType>::setIdx(StorageType idx) {
  const StorageType idMask = 0xF0000000;
  const StorageType idxMask = 0x0FFFFFFF;

  m_storage = (idMask & m_storage) | (idxMask & idx);
}

template <typename PunType>
__device__ __host__
PunType& PackedIndex<PunType>::getPun() {
  return reinterpret_cast<PunType&>(m_storage);
}

template <typename PunType>
__device__ __host__
void PackedIndex<PunType>::setPun(PunType bits) {
  reinterpret_cast<PunType&>(m_storage) = bits;
}

template <typename PunType>
__device__ __host__
typename PackedIndex<PunType>::StorageType
PackedIndex<PunType>::getUnsigned() {
  return m_storage;
}

template <typename PunType>
__device__ __host__
void PackedIndex<PunType>::setUnsigned(StorageType bits) {
  m_storage = bits;
}

} // curves
} // dubins
} // mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_CUDA2_PACKEDINDEX_HPP_
