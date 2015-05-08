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
 *  @file   Tree.hpp
 *
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  basically a c++ rewrite of http://code.google.com/p/kdtree/
 */

#ifndef MPBLOCKS_KD_TREE_TREE_HPP_
#define MPBLOCKS_KD_TREE_TREE_HPP_

#include <vector>
#include <cassert>
#include <iostream>
#include <limits>

namespace mpblocks {
namespace kd_tree {

template <class Traits>
Tree<Traits>::Tree()
    : m_root(0), m_size(0) {
  clear();
}

template <class Traits>
Tree<Traits>::~Tree() {}

template <class Traits>
void Tree<Traits>::set_initRect(const HyperRect_t& h) {
  m_initRect = h;
}

template <class Traits>
void Tree<Traits>::insert(Node_t* n) {
  m_size++;

  const Point_t& pt = n->getPoint();
  for (int i = 0; i < pt.size(); i++) {
    if (pt[i] < m_bounds.minExt[i]) m_bounds.minExt[i] = pt[i];
    if (pt[i] > m_bounds.maxExt[i]) m_bounds.maxExt[i] = pt[i];
  }

  if (m_root)
    return static_cast<NodeBase_t*>(m_root)->insert(n);
  else {
    m_root = n;
    return static_cast<NodeBase_t*>(m_root)->construct(0, 0);
  }
}

template <class Traits>
void Tree<Traits>::findNearest(const Point_t& q, NNIface_t& search) {
  if (m_root) {
    m_rect = m_bounds;
    static_cast<NodeBase_t*>(m_root)->findNearest(q, m_rect, search);
  }
}

template <class Traits>
void Tree<Traits>::findRange(RangeIface_t& search) {
  if (m_root) {
    m_rect = m_bounds;
    static_cast<NodeBase_t*>(m_root)->findRange(search, m_rect);
  }
}

template <class Traits>
typename ListBuilder<Traits>::List_t& Tree<Traits>::buildList(bool bfs) {
  m_lister.reset();
  if (bfs)
    m_lister.buildBFS(m_root);
  else
    m_lister.buildDFS(m_root);

  return m_lister.getList();
}

template <class Traits>
void Tree<Traits>::clear() {
  m_root = 0;
  m_size = 0;
  m_bounds = m_initRect;
}

template <class Traits>
int Tree<Traits>::size() {
  return m_size;
}

}  // namespace kd_tree
}  // namespace mpblocks

#endif /* CKDTREE_H_ */
