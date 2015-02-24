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

#ifndef MPBLOCKS_RED_BLACK_TREE_H_
#define MPBLOCKS_RED_BLACK_TREE_H_

#include <mpblocks/redblack/color.h>
#include <mpblocks/redblack/iterator.h>

namespace mpblocks {
namespace redblack {

/// implements red black trees from CLRS
template <class Traits>
class Tree {
 public:
  typedef typename Traits::NodeRef NodeRef;
  typedef typename Traits::NodeOps NodeOps;
  typedef typename Traits::Key Key;

  static const Color RED = Color::RED;
  static const Color BLACK = Color::BLACK;

 private:
  NodeRef Nil;
  NodeRef root;

  size_t m_size;
  NodeOps m_ops;

  Color&    color(NodeRef N){ return m_ops. color(N); }
  NodeRef&      p(NodeRef N){ return m_ops.parent(N); }
  NodeRef&   left(NodeRef N){ return m_ops.  left(N); }
  NodeRef&  right(NodeRef N){ return m_ops. right(N); }

  void swapKey(NodeRef a, NodeRef b) { m_ops.swapKey(a, b); }

 public:
  Tree(NodeRef Nil) : Nil(Nil), root(Nil), m_size(0) {}

  void clear() {
    root = Nil;
    m_size = 0;
  }

  auto key(NodeRef N) -> decltype(m_ops.key(N)) { return m_ops.key(N); }

  void leftRotate(NodeRef x) {
    NodeRef y = right(x);  //< set y
    right(x) = left(y);    //< turn y's left subtree into x's right
                           //  subtree
    p(left(y)) = x;
    p(y) = p(x);

    if (p(x) == Nil)
      root = y;
    else if (x == left(p(x)))
      left(p(x)) = y;
    else
      right(p(x)) = y;

    left(y) = x;  //< put x on y's left
    p(x) = y;
  }

  void rightRotate(NodeRef x) {
    NodeRef y = left(x);  //< set y
    left(x) = right(y);   //< turn y's left subtree into x's left
                          //  subtree
    p(right(y)) = x;
    p(y) = p(x);

    if (p(x) == Nil)
      root = y;
    else if (x == right(p(x)))
      right(p(x)) = y;
    else
      left(p(x)) = y;

    right(y) = x;  //< put x on y's right
    p(x) = y;
  }

  void insertFixup(NodeRef z) {
    while (color(p(z)) == RED) {
      if (p(z) == left(p(p(z)))) {
        NodeRef y = right(p(p(z)));
        if (color(y) == RED) {
          color(p(z)) = BLACK;
          color(y) = BLACK;
          color(p(p(z))) = RED;
          z = p(p(z));
        } else {
          if (z == right(p(z))) {
            z = p(z);
            leftRotate(z);
          }
          color(p(z)) = BLACK;
          color(p(p(z))) = RED;
          rightRotate(p(p(z)));
        }
      } else {
        NodeRef y = left(p(p(z)));
        if (color(y) == RED) {
          color(p(z)) = BLACK;
          color(y) = BLACK;
          color(p(p(z))) = RED;
          z = p(p(z));
        } else {
          if (z == left(p(z))) {
            z = p(z);
            rightRotate(z);
          }
          color(p(z)) = BLACK;
          color(p(p(z))) = RED;
          leftRotate(p(p(z)));
        }
      }
    }
    color(root) = BLACK;
  }

  void insert(NodeRef z) {
    ++m_size;

    NodeRef y = Nil;
    NodeRef x = root;
    while (x != Nil) {
      y = x;
      if (key(z) < key(x))
        x = left(x);
      else
        x = right(x);
    }
    p(z) = y;
    if (y == Nil)
      root = z;
    else if (key(z) < key(y))
      left(y) = z;
    else
      right(y) = z;

    left(z) = Nil;
    right(z) = Nil;
    color(z) = RED;
    insertFixup(z);
  }

  void removeFixup(NodeRef x) {
    while (x != root && color(x) == BLACK) {
      if (x == left(p(x))) {
        NodeRef w = right(p(x));
        if (color(w) == RED) {
          color(w) = BLACK;
          color(p(x)) = RED;
          leftRotate(p(x));
          w = right(p(x));
        }

        if (color(left(w)) == BLACK && color(right(w)) == BLACK) {
          color(w) = RED;
          x = p(x);
        } else {
          if (color(right(w)) == BLACK) {
            color(left(w)) = BLACK;
            color(w) = RED;
            rightRotate(w);
            w = right(p(x));
          }
          color(w) = color(p(x));
          color(p(x)) = BLACK;
          color(right(w)) = BLACK;
          leftRotate(p(x));
          x = root;
        }
      } else {
        NodeRef w = left(p(x));
        if (color(w) == RED) {
          color(w) = BLACK;
          color(p(x)) = RED;
          rightRotate(p(x));
          w = left(p(x));
        }

        if (color(right(w)) == BLACK && color(left(w)) == BLACK) {
          color(w) = RED;
          x = p(x);
        } else {
          if (color(left(w)) == BLACK) {
            color(right(w)) = BLACK;
            color(w) = RED;
            leftRotate(w);
            w = left(p(x));
          }
          color(w) = color(p(x));
          color(p(x)) = BLACK;
          color(left(w)) = BLACK;
          rightRotate(p(x));
          x = root;
        }
      }
    }
    color(x) = BLACK;
  }

  /// CLRS 12.2 page 258
  NodeRef treeMinimum(NodeRef x) {
    while (left(x) != Nil) x = left(x);
    return x;
  }

  /// CLRS 12.2 page 258
  NodeRef treeMaximum(NodeRef x) {
    while (right(x) != Nil) x = right(x);
    return x;
  }

  /// CLRS 12.2 page 259
  NodeRef treeSuccessor(NodeRef x) {
    if (right(x) != Nil) return treeMinimum(right(x));
    NodeRef y = p(x);
    while (y != Nil && x == right(y)) {
      x = y;
      y = p(y);
    }
    return y;
  }

  /// CLRS 12.2 page 259
  NodeRef treePredecessor(NodeRef x) {
    if (left(x) != Nil) return treeMaximum(left(x));
    NodeRef y = p(x);
    while (y != Nil && x == left(y)) {
      x = y;
      y = p(y);
    }
    return y;
  }

  NodeRef remove(NodeRef z) {
    --m_size;

    NodeRef x, y;

    if (left(z) == Nil || right(z) == Nil)
      y = z;
    else
      y = treeSuccessor(z);

    if (left(y) != Nil)
      x = left(y);
    else
      x = right(y);

    p(x) = p(y);
    if (p(y) == Nil)
      root = x;
    else if (y == left(p(y)))
      left(p(y)) = x;
    else
      right(p(y)) = x;

    if (y != z) swapKey(z, y);
    if (color(y) == BLACK) removeFixup(x);
    return y;
  }

  Iterator<Traits> begin() { return Iterator<Traits>(this, treeMinimum(root)); }

  Iterator<Traits> end() { return Iterator<Traits>(this, Nil); }

  size_t size() { return m_size; }
};

} //< namespace redblack
} //< namespace mpblocks

#endif // MPBLOCKS_RED_BLACK_TREE_H_
