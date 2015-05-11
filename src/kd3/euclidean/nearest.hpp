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
 *  @file   mpblocks/kd_tree/euclidean/Nearest.hpp
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_NEAREST_HPP_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_NEAREST_HPP_

namespace mpblocks {
namespace kd_tree {
namespace euclidean {

template <class Traits>
void Nearest<Traits>::reset(Format_t inf) {
  m_d2Best = inf;
  m_nBest = 0;
}

template <class Traits>
typename Traits::Node* Nearest<Traits>::result() {
  return m_nBest;
}

template <class Traits>
void Nearest<Traits>::evaluate(const Point_t& q, const Point_t& p, Node_t* n) {
  Format_t d2 = m_dist2Fn(q, p);
  if (d2 < m_d2Best) {
    m_d2Best = d2;
    m_nBest = n;
  }
}

template <class Traits>
bool Nearest<Traits>::shouldRecurse(const Point_t& q, const HyperRect_t& r) {
  Format_t d2 = m_dist2Fn(q, r);
  return (d2 < m_d2Best);
}

}  // namespace euclidean
}  // namespace kd_tree
}  // namespace mpblocks

#endif  // NEARESTNEIGHBOR_H_
