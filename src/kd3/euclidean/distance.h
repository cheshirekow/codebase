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

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_DISTANCE_H_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_DISTANCE_H_

namespace mpblocks {
namespace kd_tree {
namespace euclidean {

/// provides euclidean distance computation
template <class Traits>
class Distance {
 public:
  typedef typename Traits::Format_t Format_t;
  typedef Eigen::Matrix<Format_t, Traits::NDim, 1> Point_t;
  typedef HyperRect<Traits> Hyper_t;

 public:
  /// return the euclidean distance between two points
  Format_t operator()(const Point_t& pa, const Point_t& pb);

  /// return the euclidean distance between a point and hyper rectangle
  Format_t operator()(const Point_t& p, const Hyper_t& h);
};

}  // namespace eucliean
}  // namespace kd_tree
}  // namespace mpblocks

#endif  // DEFAULTDISTANCE_H_
