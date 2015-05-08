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
 *  @file   mpblocks/kd_tree/euclidean/KNearest.hpp
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_KNEAREST_HPP_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_KNEAREST_HPP_

namespace mpblocks {
namespace kd_tree {
namespace euclidean {

template <class Traits, template <class> class Allocator>
KNearest<Traits, Allocator>::KNearest(unsigned int k) {
  m_k = k;
}

template <class Traits, template <class> class Allocator>
void KNearest<Traits, Allocator>::reset() {
  m_queue.clear();
}

template <class Traits, template <class> class Allocator>
void KNearest<Traits, Allocator>::reset(int k) {
  m_queue.clear();
  m_k = k;
}

template <class Traits, template <class> class Allocator>
const typename KNearest<Traits, Allocator>::PQueue_t&
KNearest<Traits, Allocator>::result() {
  return m_queue;
}

template <class Traits, template <class> class Allocator>
void KNearest<Traits, Allocator>::evaluate(const Point_t& q, const Point_t& p,
                                           Node_t* n) {
  Key_t key;
  key.d2 = m_dist2Fn(q, p);
  key.n = n;

  m_queue.insert(key);

  if (m_queue.size() > m_k) m_queue.erase(--m_queue.end());
}

template <class Traits, template <class> class Allocator>
bool KNearest<Traits, Allocator>::shouldRecurse(const Point_t& q,
                                                const HyperRect_t& r) {
  Format_t d2 = m_dist2Fn(q, r);
  return (d2 < m_queue.rbegin()->d2 || m_queue.size() < m_k);
}

}  // namespace euclidean
}  // namespace kd_tree
}  // namespace mpblocks

#endif  // NEARESTNEIGHBOR_H_
