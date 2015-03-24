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
 *  @file   /home/josh/Codes/cpp/mpblocks2/btps/include/mpblocks/btps/Node.h
 *
 *  @date   Aug 4, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_BTPS_NODE_H_
#define MPBLOCKS_BTPS_NODE_H_

#include <cstdint>

namespace mpblocks {
namespace     btps {

/// A node in a btps tree
template <typename Traits>
struct BasicNode {
  typedef typename Traits::NodeRef NodeRef;

  uint32_t count;    ///< count of subtree descendants
  double weight;     ///< the weight of this node
  double cumweight;  ///< cumulative weight of children
  NodeRef parent;    ///< only needed if we want removal
  NodeRef left;      ///< left child
  NodeRef right;     ///< right child

  BasicNode(double weight = 0) : count(0), weight(weight), cumweight(weight) {}
};

}  //< namespace btps
}  //< namespace mpblocks

#endif  // NODE_H_
