/*
 *  Copyright (C) 2011 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of kd3.
 *
 *  kd3 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  kd3 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with kd3.  If not, see <http://www.gnu.org/licenses/>.
 */ /**
 *  @file
 *
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef KD3_EUCLIDEAN_HYPERRECT_HPP_
#define KD3_EUCLIDEAN_HYPERRECT_HPP_

#include <limits>
#include <kd3/euclidean/hyperrect.h>

namespace kd3 {
namespace euclidean {

template <class Traits>
HyperRect<Traits>::HyperRect() {
  min_ext_.fill(0);
  max_ext_.fill(0);
}

template <class Traits>
typename Traits::Scalar HyperRect<Traits>::GetSquaredDistanceTo(
    const Point& point) {
  Scalar dist2 = 0;
  Scalar dist2i = 0;

  for (unsigned int i = 0; i < point.rows(); i++) {
    if (point[i] < min_ext_[i])
      dist2i = min_ext_[i] - point[i];
    else if (point[i] > max_ext_[i])
      dist2i = max_ext_[i] - point[i];
    else
      dist2i = 0;

    dist2i *= dist2i;
    dist2 += dist2i;
  }

  return dist2;
}

template <class Traits>
typename Traits::Scalar HyperRect<Traits>::GetMeasure() {
  Scalar s = 1.0;
  for (unsigned int i = 0; i < min_ext_.rows(); i++) {
    s *= max_ext_[i] - min_ext_[i];
  }
  return s;
}

}  // namespace euclidean
}  // namespace kd3

#endif  // KD3_EUCLIDEAN_HYPERRECT_HPP_
