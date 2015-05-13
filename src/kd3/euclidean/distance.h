/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
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
 *  @file   mpblocks/kd_tree/euclidean/Distance.h
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef KD3_EUCLIDEAN_DISTANCE_H_
#define KD3_EUCLIDEAN_DISTANCE_H_

#include <cmath>
#include <Eigen/Dense>
#include <kd3/hyperrect.h>

namespace kd3 {
namespace euclidean {

/// provides euclidean distance computation
template <typename Scalar, int ndim_>
class SquaredDistance {
 public:
  typedef Eigen::Matrix<Scalar, ndim_, 1> Point;
  typedef kd3::HyperRect<Scalar, ndim_> HyperRect;

 public:
  /// return the euclidean distance between two points
  Scalar operator()(const Point& pa, const Point& pb) {
    return (pa - pb).squaredNorm();
  }

  /// return the euclidean distance between a point and hyper rectangle
  Scalar operator()(const Point& p, const HyperRect& h) {
    Scalar dist2 = 0;
    Scalar dist2i = 0;

    for (unsigned int i = 0; i < p.rows(); i++) {
      if (p[i] < h.min_ext[i]) {
        dist2i = h.min_ext[i] - p[i];
      } else if (p[i] > h.max_ext[i]) {
        dist2i = h.max_ext[i] - p[i];
      } else {
        dist2i = 0;
      }
      dist2 += dist2i * dist2i;
    }

    return dist2;
  }
};

/// provides euclidean distance computation
template <typename Scalar, int ndim_>
class Distance {
 public:
  typedef Eigen::Matrix<Scalar, ndim_, 1> Point;
  typedef kd3::HyperRect<Scalar, ndim_> HyperRect;

 public:
  /// return the euclidean distance between two points
  Scalar operator()(const Point& pa, const Point& pb) {
    return (pa - pb).norm();
  }

  /// return the euclidean distance between a point and hyper rectangle
  Scalar operator()(const Point& p, const HyperRect& h) {
    Scalar dist2 = 0;
    Scalar dist2i = 0;

    for (unsigned int i = 0; i < p.rows(); i++) {
      if (p[i] < h.min_ext[i]) {
        dist2i = h.min_ext[i] - p[i];
      } else if (p[i] > h.max_ext[i]) {
        dist2i = h.max_ext[i] - p[i];
      } else {
        dist2i = 0;
      }
      dist2 += dist2i * dist2i;
    }

    return std::sqrt(dist2);
  }
};


}  // namespace eucliean
}  // namespace kd3

#endif  // KD3_EUCLIDEAN_DISTANCE_H_
