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
 *  along with Fontconfigmm.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Aug 26, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef MPBLOCKS_BROWN79_INVERSION_HPP_
#define MPBLOCKS_BROWN79_INVERSION_HPP_

namespace  mpblocks {
namespace   brown79 {

template <class Traits>
Inversion<Traits>::Inversion() {}

template <class Traits>
Inversion<Traits>::Inversion(const Point& center, Scalar radius) {
  init(center, radius);
}

template <class Traits>
void Inversion<Traits>::init(const Point& center, Scalar radius) {
  m_center = center;
  m_radius = radius;
}

template <class Traits>
typename Traits::Point Inversion<Traits>::invert(const Point& p) {
  // subtract from center ot get a vector
  Point v = p - m_center;

  // now invert vector
  v = (m_radius * m_radius / v.squaredNorm()) * v;

  // and add back to the center to get the result point
  return v + m_center;
}

template <class Traits>
typename Inversion<Traits>::Point Inversion<Traits>::operator()(
    const Point& p) {
  return invert(p);
}

template <class Traits>
const typename Traits::Point& Inversion<Traits>::center() const {
  return m_center;
}

template <class Traits>
typename Traits::Scalar Inversion<Traits>::radius() const {
  return m_radius;
}

} // namespace brown79 
} // namespace mpblocks 


#endif // MPBLOCKS_BROWN79_INVERSION_HPP_
