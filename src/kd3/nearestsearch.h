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
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_NEARESTSEARCH_H_
#define MPBLOCKS_KD_TREE_NEARESTSEARCH_H_

namespace mpblocks {
namespace kd_tree {

/// Interface for nearest node type searches
template <class Traits>
struct NearestSearchIface {
  typedef typename Traits::Format_t Format_t;
  typedef typename Traits::Node Node_t;
  typedef typename Traits::HyperRect HyperRect_t;

  typedef Eigen::Matrix<Format_t, Traits::NDim, 1> Vector_t;
  typedef Vector_t Point_t;

  /// just ensure virtualness
  virtual ~NearestSearchIface() {}

  /// evaluate the current point, and add the corresponding node to the
  /// result set if necessary
  virtual void evaluate(const Point_t& q, const Point_t& p, Node_t* n) = 0;

  /// evaluate the hyper rectangle and return true if it is possible that
  /// we must recurse into it
  virtual bool shouldRecurse(const Point_t& q, const HyperRect_t& h) = 0;
};

}  // namespace kd_tree
}  // namespace mpblocks

#endif
