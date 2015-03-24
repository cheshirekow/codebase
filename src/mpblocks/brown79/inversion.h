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
 *  @date   Aug 26, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_BROWN79_INVERSION_H_
#define MPBLOCKS_BROWN79_INVERSION_H_

namespace mpblocks {
namespace brown79 {

/// Represents the inversion operator described in Brown.
/**
 *  An inversion has both a center and a radius. The inversion operator is
 *  self involute in the sense if the double image of the inversion is the
 *  original object.
 *
 *  The inverse x' of a point x is found by (x' - c) = r / |x-c|^2 (x-c).
 *  i.e. it lies on the ray from c->x and the distance from c is r / m where
 *  m is the distance from c to x, r is the radius of the inversion, and
 *  c is the center of inversion
 */
template <class Traits>
class Inversion {
 public:
  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::Point Point;

 private:
  Point m_center;   ///< the center of inversion
  Scalar m_radius;  ///< the radius of inversion

 public:
  /// create an uninitialized inversion
  Inversion();

  /// create a new inversion map with specified center and radius
  Inversion(const Point& center, Scalar radius = 1.0);

  /// alter the inversion map to have a new center and radius
  void init(const Point& center, Scalar radius = 1.0);

  /// invert a point
  Point invert(const Point& p);

  /// invert a point
  Point operator()(const Point& p);

  /// return the center of this inversion map
  const Point& center() const;

  /// return the radius of this inversion map
  Scalar radius() const;
};

} // namespace brown79 
} // namespace mpblocks 

#endif // INVERSION_H_
