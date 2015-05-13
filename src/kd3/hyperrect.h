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

#ifndef KD3_HYPERRECT_H_
#define KD3_HYPERRECT_H_

#include <cstdint>
#include <Eigen/Dense>

namespace kd3 {

/// an NDim dimensional hyperrectangle, represented as a min and max extent
/**
 *  implemented by storing the minimum and maximum extent of the hyper-rectangle
 *  in all dimensions
 */
template <typename Scalar, int ndim_>
struct HyperRect {
  typedef Eigen::Matrix<Scalar, ndim_, 1> Point;

  Point min_ext;  ///< minimum extent of hyper-rectangle
  Point max_ext;  ///< maximum extent of hyper-rectangle

  /// initialize min and max ext to be 0,0,...
  HyperRect() {
    min_ext.fill(0);
    max_ext.fill(0);
  }

  HyperRect(const Point& min_ext_in, const Point& max_ext_in)
      : min_ext(min_ext_in), max_ext(max_ext_in) {}

  /// return the measure (volume) of the hyperrectangle
  Scalar GetMeasure() const {
    Scalar s = 1.0;
    for (unsigned int i = 0; i < min_ext.rows(); i++) {
      s *= max_ext[i] - min_ext[i];
    }
    return s;
  }

  /// return the length of a particular axis
  Scalar GetLength(uint8_t axis) const { return max_ext[axis] - min_ext[axis]; }

  /// return the index of the longest dimension
  uint8_t GetLongestDimension() const {
    uint8_t i_max = 0;
    for (uint8_t i = 1; i < ndim_; i++) {
      if (GetLength(i) > GetLength(i_max)) {
        i_max = i;
      }
    }
  }

  /// Split the hyperrectangle on one axis at the specified value, making this
  /// hyprerectangle refer to the half with values smaller than the split
  void SplitLesser(uint8_t dim, Scalar value) { max_ext[dim] = value; }

  /// Split the hyperrectangle on one axis at the specified value, making this
  /// hyprerectangle refer to the half with values larger than the split
  void SplitGreater(uint8_t dim, Scalar value) { min_ext[dim] = value; }

  /// split the hyperrectangle at the given point along the given dimension
  void Split(uint8_t dim, Scalar value, HyperRect<Scalar, ndim_>* left,
             HyperRect<Scalar, ndim_>* right) const {
    if (left) {
      *left = *this;
      left->max_ext[dim] = value;
    }
    if (right) {
      *right = *this;
      right->min_ext[dim] = value;
    }
  }

  void GrowToContain(const Point& point) {
    for (uint8_t i = 0; i < ndim_; i++) {
      if (point[i] < min_ext[i]) {
        min_ext[i] = point[i];
      }
      if (point[i] > max_ext[i]) {
        max_ext[i] = point[i];
      }
    }
  }
};

}  // namespace kd3

#endif  // KD3_HYPERRECT_H_
