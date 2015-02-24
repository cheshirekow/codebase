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
 *  @file
 *  @date   Jun 20, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA_PACKEDINDEX_H_
#define MPBLOCKS_DUBINS_CURVES_CUDA_PACKEDINDEX_H_

// Don't include <cstdint> as that requires c++11 and nvcc does not yet
// support that. It doest support c99 hover, co stdint.h is OK.
#include <stdint.h>
#include <mpblocks/dubins/curves/types.h>
#include <mpblocks/dubins/curves_cuda/portable.h>

namespace    mpblocks {
namespace      dubins {
namespace curves_cuda {

/// Template meta function which defines internal typedef Result to be an
/// unsigned integer of size BYTES
template <unsigned int BYTES>
struct PackedStorage{ typedef uint32_t Result; };

template <> struct PackedStorage<1>{ typedef uint8_t  Result; };
template <> struct PackedStorage<2>{ typedef uint16_t Result; };
template <> struct PackedStorage<4>{ typedef uint32_t Result; };
template <> struct PackedStorage<8>{ typedef uint64_t Result; };

/// stores both the index of the entry and the solution id of it's best solver
/// in a bitfield of the same size as PunType and provides conversion to and
/// from PunType
template <typename PunType>
class PackedIndex {
 public:
  typedef typename PackedStorage<sizeof(PunType)>::Result StorageType;

 private:
  StorageType m_storage;

 public:
  __device__ __host__
  PackedIndex(PunType bits);

  __device__ __host__
  PackedIndex(SolutionId id = INVALID, StorageType idx = 0);

  __device__ __host__
  SolutionId getId() const;

  __device__ __host__
  StorageType getIdx() const;

  __device__ __host__
  void setId(SolutionId id);

  __device__ __host__
  void setIdx(StorageType idx);

  __device__ __host__
  PunType& getPun();

  __device__ __host__
  void setPun(PunType bits);

  __device__ __host__
  StorageType getUnsigned();

  __device__ __host__
  void setUnsigned(StorageType bits);
};

}  // curves
}  // dubins
}  // mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_CUDA_PACKEDINDEX_H_
