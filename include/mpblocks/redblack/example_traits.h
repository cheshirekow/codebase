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

#ifndef MPBLOCKS_RED_BLACK_EXAMPLETRAITS_H_
#define MPBLOCKS_RED_BLACK_EXAMPLETRAITS_H_

#include <mpblocks/redblack/color.h>
#include <mpblocks/redblack/node.h>
#include <utility>

namespace mpblocks {
namespace redblack {

/// an example traits struction from which a red-black tree may be
/// instantiated
struct ExampleTraits {
  /// the key type, must be comparable, note that this typedef is not
  /// required, the tree will operate on whatever NodeOps::key returns,
  /// so that has to be comparable, but you do not need to explicitly
  /// indicate the type here
  typedef double Key;

  /// this type is not required by the interface, but if you just need a
  /// simple node type then this one will do it for you
  typedef BasicNode<ExampleTraits> Node;

  /// some type which stores uniquely identifies nodes, for instance
  /// a node pointer or an index into an array
  typedef Node* NodeRef;

  /// a callable type which implements the primitives required to access
  /// fields of a node
  struct NodeOps {
    /// return the color of node N
    Color&      color( NodeRef N ){ return N->color; }

    /// return the parent node of N
    NodeRef&   parent( NodeRef N ){ return N->parent; }

    /// return the left child of N
    NodeRef&     left( NodeRef N ){ return N->left; }

    /// return the right child of N
    NodeRef&    right( NodeRef N ){ return N->right; }

    /// return the key associated with N
    Key&          key( NodeRef N ){ return N->key; }

    /// swap the keys in two nodes
    void swapKey(NodeRef a, NodeRef b) { std::swap(a->key, b->key); }
};
};

} //< namespace redblack
} //< namespace mpblocks

#endif  // MPBLOCKS_RED_BLACK_EXAMPLETRAITS_H_
