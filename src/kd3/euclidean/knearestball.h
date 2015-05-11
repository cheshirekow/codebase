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
 *  @file   mpblocks/kd_tree/euclidean/KNearestBall.h
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_KNEARESTBALL_H_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_KNEARESTBALL_H_

#include <set>

namespace mpblocks {
namespace kd_tree {
namespace euclidean {

template <class Traits, template <class> class Allocator = std::allocator>
class KNearestBall : public NearestSearchIface<Traits> {
 public:
  typedef typename Traits::Format_t Format_t;
  typedef typename Traits::Node Node_t;

  typedef Distance<Traits> Distance_t;
  typedef HyperRect<Traits> HyperRect_t;
  typedef Key<Traits> Key_t;
  typedef typename Key_t::Compare KeyCompare_t;
  typedef Allocator<Key_t> Allocator_t;

  typedef Eigen::Matrix<Format_t, Traits::NDim, 1> Point_t;
  typedef std::set<Key_t, KeyCompare_t, Allocator_t> PQueue_t;

 protected:
  unsigned int m_k;
  Format_t m_radius;
  PQueue_t m_queue;
  Distance_t m_dist2Fn;

 public:
  KNearestBall(Format_t radius = 1, unsigned int k = 1);

  virtual ~KNearestBall(){};

  // clear the queue
  void reset();

  // clear the queue and change k
  void reset(Format_t radius, int k);

  // return the result
  const PQueue_t& result();

  /// calculates Euclidean distance from @p q to @p p, and if its less
  /// than the current best replaces the current best node with @p n
  virtual void evaluate(const Point_t& q, const Point_t& p, Node_t* n);

  /// evaluate the Euclidean distance from @p q to it's closest point in
  /// @p r and if that distance is less than the current best distance,
  /// return true
  virtual bool shouldRecurse(const Point_t& q, const HyperRect_t& r);
};

}  // namespace euclidean
}  // namespace kd_tree
}  // namespace mpblocks

#endif  // NEARESTNEIGHBOR_H_
