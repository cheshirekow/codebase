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
 *  @date   Jul 3, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_SPECPACK_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_SPECPACK_H_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {
namespace        hyper {

typedef unsigned int uint;
const uint INVALID_SPEC = 0x01 << 5;

template <int xSpec, int ySpec, int tSpec>
struct SpecPack {
  static const uint Result = (xSpec << 4) | (ySpec << 2) | (tSpec);
};

template <int xSpec, int ySpec, int tSpec>
const uint SpecPack<xSpec, ySpec, tSpec>::Result;

inline uint packSpec(int xSpec, int ySpec, int tSpec) {
  return (xSpec << 4) | (ySpec << 2) | (tSpec);
}

inline uint boolToSpec(bool min, bool max) {
  if (min)
    return MIN;
  else if (max)
    return MAX;
  else
    return OFF;
}

} // namespace hyper
} // namespace curves_eigen
} // namespace dubins
} // namespace mpblocks

#endif // MPBLOCKS_DUBINS_CURVES_EIGEN_SPECPACK_H_
