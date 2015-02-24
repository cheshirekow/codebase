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
 *  @file   /home/josh/Codes/cpp/mpblocks2/dubins/include/mpblocks/dubins/curves/Result.h
 *
 *  @date   Jun 26, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_RESULT_H_
#define MPBLOCKS_DUBINS_CURVES_RESULT_H_

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace mpblocks {
namespace   dubins {

/// Encapsulates the solution distance along with a feasibility bit for a
/// particular primitive solution
template <typename Format_t>
struct Result {
  Format_t d;  ///< distance
  bool f;      ///< is feasible

  /// the default constructor is for infeasible
  __device__ __host__
  Result() : d(0), f(false) {}

  /// we only use this constructor when it's feasible
  __device__ __host__
  Result(Format_t d, bool f = true) : d(d), f(f) {}

  /// because I'm lazy
  __device__ __host__
  operator Format_t() const { return d; }

  __device__ __host__
  operator bool() const { return f; }

  __device__ __host__
  Result<Format_t>& operator=(Format_t d_in) {
    d = d_in;
    f = true;
    return *this;
  }

  __device__ __host__
  Result<Format_t>& operator=(
      const Result<Format_t>& other) {
    d = other.d;
    f = other.f;
    return *this;
  }
};

/// Encapsulates a solution distance along with the id of the path type,
/// identifying the nature of the three arc segments in the path.
template <typename Format_t>
struct DistanceAndId {
  Format_t d;  ///< distance
  int id;      ///< id

  __device__ __host__
  DistanceAndId() : d(0), id(INVALID) {}

  __device__ __host__
  DistanceAndId(Format_t d, SolutionId id) : d(d), id(id) {}

  /// can act like a bool
  __device__ __host__
  operator bool() const { return (id != INVALID); }

  /// set the storage and make the flag true
  __device__ __host__
  void set(Format_t d_in, int id_in) {
    d = d_in;
    id = id_in;
  }

  __device__ __host__
  Result<Format_t>& operator=(
      const Result<Format_t>& other) {
    id = other.id;
    d = other.d;
    return *this;
  }
};

} // dubins
} // mpblocks

#endif // MPBLOCKS_DUBINS_CURVES_RESULT_H_
