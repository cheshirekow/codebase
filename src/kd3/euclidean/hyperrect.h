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
 */
/**
 *  @file
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef KD3_EUCLIDEAN_HYPERRECT_H_
#define KD3_EUCLIDEAN_HYPERRECT_H_

#include <Eigen/Dense>

namespace kd3 {
namespace euclidean {

/// an NDim dimensional hyperrectangle, represented as a min and max extent
/**
 *  implemented by storing the minimum and maximum extent of the hyper-rectangle
 *  in all dimensions
 */
template <class Traits>
struct HyperRect {
  typedef typename Traits::Scalar Scalar;
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> Point;

  Point min_ext_;  ///< minimum extent of hyper-rectangle
  Point max_ext_;  ///< maximum extent of hyper-rectangle

  /// initialize min and max ext to be 0,0,...
  HyperRect();

  /// find the nearest point in the hyper-rectangle to the query point and
  /// return it's distance (squared)
  Scalar GetSquaredDistanceTo(const Point& point);

  /// return the measure (volume) of the hyperrectangle
  Scalar GetMeasure();
};

}  // namespace euclidean
}  // namespace kd3

#endif  // KD3_EUCLIDEAN_HYPERRECT_H_
