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

#ifndef MPBLOCKS_RED_BLACK_NODE_H_
#define MPBLOCKS_RED_BLACK_NODE_H_

#include <mpblocks/redblack/color.h>
#include <cstdint>

namespace mpblocks {
namespace redblack {

/// A node in a redblack tree
template <typename Traits>
struct BasicNode {
  typedef typename Traits::Key Key;
  typedef typename Traits::NodeRef NodeRef;

  Color color;
  Key key;
  NodeRef parent;
  NodeRef left;
  NodeRef right;

  BasicNode(Key key, NodeRef Nil)
      : color(Color::BLACK), key(key), parent(Nil), left(Nil), right(Nil) {}

  BasicNode(NodeRef Nil)
      : color(Color::BLACK), parent(Nil), left(Nil), right(Nil) {}
};

} //< namespace redblack
} //< namespace mpblocks

#endif  // MPBLOCKS_RED_BLACK_NODE_H_
