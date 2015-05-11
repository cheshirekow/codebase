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
 *  @file
 *
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef KD3_NEARESTSEARCH_H_
#define KD3_NEARESTSEARCH_H_

#include <Eigen/Dense>

namespace kd3 {

/// Interface for nearest node type searches
template <class Traits>
struct NearestSearchIface {
  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::Node Node;
  typedef typename Traits::HyperRect HyperRect;

  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> Vector;
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> Point;

  /// just ensure virtualness
  virtual ~NearestSearchIface() {}

  /// evaluate the current point, and add the corresponding node to the
  /// result set if necessary
  virtual void Evaluate(const Point& q, const Point& p, Node* n) = 0;

  /// evaluate the hyper rectangle and return true if it is possible that
  /// we must recurse into it
  virtual bool ShouldRecurse(const Point& q, const HyperRect& h) = 0;
};

}  // namespace kd3

#endif  // KD3_NEARESTSEARCH_H_
