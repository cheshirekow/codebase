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
 *  @file   /home/josh/Codes/cpp/mpblocks2/btps/include/mpblocks/btps/ExampleTraits.h
 *
 *  @date   Aug 4, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_BTPS_EXAMPLETRAITS_H_
#define MPBLOCKS_BTPS_EXAMPLETRAITS_H_

#include <mpblocks/btps/node.h>
#include <cstdint>

namespace mpblocks {
namespace     btps {

/// an example traits structure from which a balanced tree of partial sums
/// may be instantiated
struct ExampleTraits {
  struct Node;

  /// some type which stores uniquely identifies nodes, for instance
  /// a node pointer or an index into an array
  typedef Node* NodeRef;

  /// this type is not required by the interface, but if you just need a
  /// simple node type then this one will do it for you
  struct Node : BasicNode<ExampleTraits> {
    Node(double weight = 0) : BasicNode<ExampleTraits>(weight), freq(0) {}

    uint32_t freq;  ///< number of times sampled
  };

  /// a callable type which implements the primitives required to access
  /// fields of a node
  struct NodeOps {
    /// you can leave this one off you if you dont need removal
    NodeRef& Parent(NodeRef N) { return N->parent; }

    /// return the left child of N
    NodeRef& LeftChild(NodeRef N) { return N->left; }

    /// return the right child of N
    NodeRef& RightChild(NodeRef N) { return N->right; }

    /// return the weight of this node in particular
    double Weight(NodeRef N) { return N->weight; }

    /// return the cumulative weight of the subtree, the return type
    /// is deduced by the tree template and can be anything modeling a
    /// real number
    double& CumulativeWeight(NodeRef N) { return N->cumweight; }

    /// return the subtree node count
    uint32_t& Count(NodeRef N) { return N->count; }
  };
};

} //< namespace btps
} //< namespace mpblocks














#endif // EXAMPLETRAITS_H_
