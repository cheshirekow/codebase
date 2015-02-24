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
 *  @date   Aug 4, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef MPBLOCKS_RED_BLACK_ITERATOR_H_
#define MPBLOCKS_RED_BLACK_ITERATOR_H_

#include <iterator>

namespace mpblocks {
namespace redblack {

template< class Traits >
struct Tree;

/// implements red black trees from CLRS
template <class Traits>
struct Iterator {
  typedef typename Traits::NodeRef NodeRef;
  typedef Tree<Traits> Tree_t;

  Tree_t* tree;
  NodeRef node;

  /// dereference to whatever key returns
  auto operator*() -> decltype(tree->key(node)) { return tree->key(node); }

  Iterator(Tree_t* tree, NodeRef node) : tree(tree), node(node) {}

  /// increment operator
  Iterator<Traits>& operator++() {
    node = tree->treeSuccessor(node);
    return *this;
  }

  /// decrement operator
  Iterator<Traits>& operator--() {
    node = tree->treePredecessor(node);
    return *this;
  }

  /// comparison for range-based
  bool operator!=(const Iterator& other) { return node != other.node; }
};

} //< namespace redblack
} //< namespace mpblocks

#endif  // ITERATOR_H_
