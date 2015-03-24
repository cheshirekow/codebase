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
 *  @date   Nov 5, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_DUBINS_CURVES_FUNCS_HPP_
#define MPBLOCKS_DUBINS_CURVES_FUNCS_HPP_

#include <mpblocks/dubins/curves/funcs.h>

namespace mpblocks {
namespace   dubins {

template <typename Format_t>
__host__ __device__
Format_t clampRadian(Format_t a) {
  const Format_t _PI = static_cast<Format_t>(M_PI);
  const Format_t _2 = static_cast<Format_t>(2);

  while (a > _PI) a -= _2 * _PI;
  while (a < -_PI) a += _2 * _PI;

  return a;
}

/// Given two arc positions (in radians), return the arc distance from @p a to
/// @p in the counter-clockwise direction
template <typename Format_t>
__host__ __device__
Format_t ccwArc(Format_t a, Format_t b) {
  const Format_t _PI = static_cast<Format_t>(M_PI);
  const Format_t _2 = static_cast<Format_t>(2);

  Format_t r = b - a;
  while (r < 0) r += _2 * _PI;
  return r;
}

/// Given two arc positions (in radians), return the arc distance from @p a to
/// @p in the counter-clockwise direction
template <typename Format_t>
__host__ __device__
Format_t leftArc(Format_t a, Format_t b) {
  return ccwArc(a, b);
}

/// Given two arc positions (in radians), return the arc distance from @p a to
/// @p in the clockwise direction
template <typename Format_t>
__host__ __device__
Format_t cwArc(Format_t a, Format_t b) {
  const Format_t _PI = static_cast<Format_t>(M_PI);
  const Format_t _2 = static_cast<Format_t>(2);

  Format_t r = a - b;
  while (r < 0) r += _2 * _PI;
  return r;
}

/// Given two arc positions (in radians), return the arc distance from @p a to
/// @p in the clockwise direction
template <typename Format_t>
__host__ __device__
Format_t rightArc(Format_t a, Format_t b) {
  return cwArc(a, b);
}

} // dubins
} // mpblocks

#endif // MPBLOCKS_DUBINS_CURVES_FUNCS_HPP_
