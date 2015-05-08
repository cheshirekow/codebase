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
 *  @file   mpblocks/kd_tree/r2_s1/Distance.h
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_R2_S1_DISTANCE_H_
#define MPBLOCKS_KD_TREE_R2_S1_DISTANCE_H_

namespace mpblocks {
namespace kd_tree {
namespace r2_s1 {

/// provides r2_s1 distance computation
template <class Traits>
class Distance {
 public:
  typedef typename Traits::Format_t Format_t;
  typedef typename Traits::HyperRect Hyper_t;

  typedef Eigen::Matrix<Format_t, Traits::NDim, 1> Point_t;

 private:
  Format_t m_min;     ///< min interval
  Format_t m_max;     ///< max interval
  Format_t m_weight;  ///< weight of s1 in group metric,

 public:
  // default is [0,1] with weight of 1.0
  Distance();

  Distance(Format_t min, Format_t max, Format_t weight);
  void configure(Format_t min, Format_t max, Format_t weight);

  /// return the r2_s1 distance between two points
  Format_t operator()(const Point_t& pa, const Point_t& pb);

  /// return the r2_s1 distance between a point and hyper rectangle
  Format_t operator()(const Point_t& p, const Hyper_t& h);
};

}  // namespace eucliean
}  // namespace kd_tree
}  // namespace mpblocks

#endif  // DEFAULTDISTANCE_H_
