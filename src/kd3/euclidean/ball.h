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
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef KD3_EUCLIDEAN_BALLSEARCH_H_
#define KD3_EUCLIDEAN_BALLSEARCH_H_

namespace kd3 {
namespace euclidean {

template <class Traits>
class BallSearch : public RangeSearchIface<Traits> {
 public:
  typedef typename Traits::Format_t Format_t;
  typedef typename Traits::Node Node_t;

  typedef Distance<Traits> Distance_t;
  typedef HyperRect<Traits> HyperRect_t;
  typedef Key<Traits> Key_t;
  typedef typename Key_t::Compare KeyCompare_t;
  typedef Allocator<Key_t> Allocator_t;

  typedef Eigen::Matrix<Format_t, Traits::NDim, 1> Point_t;
  typedef std::vector<Node_t*, Allocator_t> List_t;

 protected:
  Point_t m_center;
  Format_t m_radius;
  List_t m_list;
  Distance_t m_dist2Fn;

 public:
  Ball(const Point_t& center = Point_t::Zero(), Format_t radius = 1);

  virtual ~Ball(){};

  // clear the queue
  void reset();

  // clear the queue and change k
  void reset(const Point_t& center, Format_t radius);

  // return the result
  const List_t& result();

  /// calculates Euclidean distance from @p q to @p p, and if its less
  /// than the current best replaces the current best node with @p n
  virtual void evaluate(const Point_t& p, Node_t* n);

  /// evaluate the Euclidean distance from @p q to it's closest point in
  /// @p r and if that distance is less than the current best distance,
  /// return true
  virtual bool shouldRecurse(const HyperRect_t& r);
};


template <class Traits, template <class> class Allocator>
Ball<Traits, Allocator>::Ball(const Point_t& center, Format_t radius) {
  reset(center, radius);
}

template <class Traits, template <class> class Allocator>
void Ball<Traits, Allocator>::reset() {
  m_list.clear();
}

template <class Traits, template <class> class Allocator>
void Ball<Traits, Allocator>::reset(const Point_t& center, Format_t radius) {
  m_center = center;
  m_radius = radius;
  m_list.clear();
}

template <class Traits, template <class> class Allocator>
const typename Ball<Traits, Allocator>::List_t&
Ball<Traits, Allocator>::result() {
  return m_list;
}

template <class Traits, template <class> class Allocator>
void Ball<Traits, Allocator>::evaluate(const Point_t& p, Node_t* n) {
  Format_t r2 = m_radius * m_radius;
  Format_t d2 = m_dist2Fn(p, m_center);

  if (d2 < r2) m_list.push_back(n);
}

template <class Traits, template <class> class Allocator>
bool Ball<Traits, Allocator>::shouldRecurse(const HyperRect_t& r) {
  // check if we have to recurse into the farther subtree by
  // finding out if the nearest point in the hyperrectangle is
  // inside the ball
  Format_t dist2 = m_dist2Fn(m_center, r);
  return (dist2 < m_radius * m_radius);
}


}  // namespace euclidean
}  // namespace kd3

#endif  // KD3_EUCLIDEAN_BALLSEARCH_H_
