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
 *
 *  @date   Oct 25, 2012
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */

#ifndef EDELSBRUNNER96_FACET_H_
#define EDELSBRUNNER96_FACET_H_

#include <array>
#include <cstdint>

namespace edelsbrunner96 {

/// A facet is a (d-1)-dimensional simplex.
/**
 *  We represent a facet by pointing to a pair of simplices. It is assumed
 *  that the simplices contain NDim common vertices, and each simlex contains
 *  one vertex which is not common to the other.
 */
template<class Traits>
struct Facet {
  typedef typename Traits::Storage Storage;
  typedef typename Traits::SimplexRef SimplexRef;

  std::array<SimplexRef, 2> s;  ///< the two simplices
  std::array<uint16_t, 2> v;   ///< the version of the two simplices at creation

  /// Assigns the two simplex references and latches their current version
  /// number
  void Construct(Storage& storage, SimplexRef s_0, SimplexRef s_1);

  /// Returns true if the two simplex references still have the same version
  /// as the time when Construct was called
  bool StillExists(Storage& storage);
};

}  // namespace edelsbrunner96

#endif // EDELSBRUNNER96_FACET_H_
