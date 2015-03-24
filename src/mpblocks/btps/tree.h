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
 *  @file   /home/josh/Codes/cpp/mpblocks2/btps/include/mpblocks/btps/Tree.h
 *
 *  @date   Aug 4, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_BTPS_TREE_H_
#define MPBLOCKS_BTPS_TREE_H_

#include <cassert>
#include <algorithm>
#include <map>
#include <list>

//#include <iostream>
//#include <boost/format.hpp>

namespace mpblocks {
namespace     btps {

/// implements a binary tree of partial sums for sampling from discrete
/// distributions with arbitrary weight
template <class Traits>
class Tree {
 public:
  typedef typename Traits::NodeRef NodeRef;
  typedef typename Traits::NodeOps NodeOps;

 private:
  NodeRef nil_;
  NodeRef root_;
  NodeOps ops_;

  NodeRef& parent(NodeRef N) { return ops_.Parent(N); }
  NodeRef& left(NodeRef N) { return ops_.LeftChild(N); }
  NodeRef& right(NodeRef N) { return ops_.RightChild(N); }

  auto count(NodeRef N) -> decltype(ops_.Count(N)) {
    return ops_.Count(N);
  }

  auto cum(NodeRef N) -> decltype(ops_.CumulativeWeight(N)) {
    return ops_.CumulativeWeight(N);
  }

  auto weight(NodeRef N) -> decltype(ops_.Weight(N)) {
    return ops_.Weight(N);
  }

 public:
  /// initialize a tree using Nil as the sentinal object
  /**
   *  Nil and NodeOps must be such that the following are true
   *
   *  op          | condition
   *  ------------|--------------
   *  weight(Nil) | = 0
   *  cum(Nil)    | = 0
   *  count(Nil)  | = 0
   *  left(Nil)   | = Nil
   *  right(Nil)  | = Nil
   *  parent(Nil) | = Nil
   *
   */
  Tree(NodeRef Nil) : nil_(Nil), root_(Nil) {
    cum(nil_) = 0;
    count(nil_) = 0;
    parent(nil_) = nil_;
    left(nil_) = nil_;
    right(nil_) = nil_;
  }

  /// return the root node in the tree
  NodeRef root() { return root_; }

  /// return the total weight of all nodes in the tree
  auto sum() -> decltype(ops_.CumulativeWeight(root_)) {
    return ops_.CumulativeWeight(root_);
  }

  /// clear out the tree (does not free nodes)
  void clear() { root_ = nil_; }

  /// return the count of nodes in the tree
  size_t size() { return count(root_); }

 private:
  /// update a node's count and cumulative weight by summing the
  /// count/weight of itself with the cumulative count/weight of
  /// it's children
  void update(NodeRef x) {
    cum(x) = weight(x) + cum(left(x)) + cum(right(x));
    count(x) = 1 + count(left(x)) + count(right(x));
  }

  /// recursively insert @p x in the subtree rooted at @p p
  void insert(NodeRef p, NodeRef x) {
    assert(p != nil_);

    // if either child is empty then we can fill that slot
    if (left(p) == nil_) {
      left(p) = x;
      parent(x) = p;
    } else if (right(p) == nil_) {
      right(p) = x;
      parent(x) = p;
    }

    // otherwise find the child with the smallest progeny
    else {
      if (count(left(p)) < count(right(p)))
        insert(left(p), x);
      else
        insert(right(p), x);
    }

    update(p);
  }

  /// walk the tree along the parent path from p to the root and
  /// update all nodes on that path
  void updateAncestry(NodeRef p) {
    if (p == nil_)
      return;
    else
      update(p);

    if (p == root_)
      return;
    else
      updateAncestry(parent(p));
  }

  /// find the deepest leaf in a subtree
  NodeRef deepestLeaf(NodeRef p) {
    if (count(p) == 1) return p;
    if (count(left(p)) > count(right(p)))
      return deepestLeaf(left(p));
    else
      return deepestLeaf(right(p));
  }

  /// true if @p x is a leaf node (i.e. both children are Nil)
  bool isLeaf(NodeRef x) { return (left(x) == nil_ && right(x) == nil_); }

  /// swaps everything but the weight and cumulative weight
  void swap(NodeRef a, NodeRef b) {
//    typedef boost::format fmt;
//    std::cout << "before swap:\n"
//              << fmt("ref:      %16u   |   %16u \n") % a % b
//              << fmt("parent:   %16u   |   %16u \n") % parent(a) % parent(b)
//              << fmt("left:     %16u   |   %16u \n") % left(a)   % left(b)
//              << fmt("right:    %16u   |   %16u \n") % right(a)  % right(b)
//              << fmt("count:    %16u   |   %16u \n") % count(a)  % count(b);

    if (parent(a) != nil_) {
      if (left(parent(a)) == a)
        left(parent(a)) = b;
      else {
        assert(right(parent(a)) == a);
        right(parent(a)) = b;
      }
    }

    if (parent(b) != nil_) {
      if (left(parent(b)) == b)
        left(parent(b)) = a;
      else {
        assert(right(parent(b)) == b);
        right(parent(b)) = a;
      }
    }

    std::swap(parent(a), parent(b));
    std::swap(left(a), left(b));
    std::swap(right(a), right(b));
    std::swap(count(a), count(b));

    if (left(a) != nil_) parent(left(a)) = a;
    if (right(a) != nil_) parent(right(a)) = a;
    if (left(b) != nil_) parent(left(b)) = b;
    if (right(b) != nil_) parent(right(b)) = b;
//
//  std::cout << "after swap:\n"
//            << fmt("ref:      %16u   |   %16u \n") % a % b
//            << fmt("parent:   %16u   |   %16u \n") % parent(a) % parent(b)
//            << fmt("left:     %16u   |   %16u \n") % left(a)   % left(b)
//            << fmt("right:    %16u   |   %16u \n") % right(a)  % right(b)
//            << fmt("count:    %16u   |   %16u \n") % count(a)  % count(b);
  }

  /// remove a leaf node from the tree
  void removeLeaf(NodeRef x) {
    assert(isLeaf(x));
    NodeRef p = parent(x);

    // if x doesn't have a parent then it must be the root so
    // clear out the tree
    if (p == nil_) {
      assert(x == root_);

      // since x is the root and it is also a leaf then we
      // must empty the tree
      root_ = nil_;
    }
    // if x has a parent, then remove x from it's list of children
    else {
      if (left(p) == x)
        left(p) = nil_;
      else
        right(p) = nil_;
      parent(x) = nil_;

      // update all the counts up to the root
      updateAncestry(p);
    }
  }

  /// given @p @$ val \in [0,1] @$, sample a node from the weighted
  /// distribution over the subtree rooted at @p x
  template <typename T>
  NodeRef findInterval(NodeRef x, T val) {
    assert(x != nil_);

    auto wLeft = cum(left(x));
    auto wMiddle = wLeft + weight(x);

    // if the value is in the left third of the split then recurse
    // on the left subtree
    if (val < wLeft) return findInterval(left(x), val);

    // if the value is in the middle of the split then we have
    // found the node to return
    else if (val < wMiddle || right(x) == nil_)
      return x;

    // otherwise, recurse on the right half, but note that we have
    // just pruned cum(left(x)) and weight(x) from the search
    // interval and since subtrees dont know their offset we have
    // to notify them by reducing the search value
    else
      return findInterval(right(x), val - wMiddle);
  }

  /// return the  larger of the error of `cum(x)` or the error of it's
  /// children
  double validateNode(NodeRef x) {
    if (x == nil_) return 0;

    double err = cum(x) - (cum(left(x)) + weight(x) + cum(right(x)));
    err *= err;

    err = std::max(err, validateNode(left(x)));
    err = std::max(err, validateNode(right(x)));
    return err;
  }

 public:
  /// insert a single node into the tree, `weight(x)` must be set, all
  /// other fields are initialized by the tree
  void insert(NodeRef x) {
    parent(x) = nil_;
    left(x) = nil_;
    right(x) = nil_;
    count(x) = 1;
    cum(x) = weight(x);

    if (root_ == nil_)
      root_ = x;
    else
      insert(root_, x);
  }

  /// remove node from the tree and maintain balance
  void remove(NodeRef x) {
    // if x is a leaf, then simply remove it
    if (isLeaf(x)) removeLeaf(x);

    // if x is not a leaf then we need to swap it out with a leaf
    // so that we can remove it without rebalancing the tree
    else {
      // find the deepest leaf in the tree to replace x
      NodeRef r = deepestLeaf(root_);

      // swap x with r
      swap(x, r);

      // if x was the root we need to swap the pointer to r
      if (root_ == x) root_ = r;

      // update the ancestry of r up to the root
      updateAncestry(r);

      // and now remove x
      removeLeaf(x);
    }
  }

  /// return the largest difference between `cum(x)` and the computed
  /// value over all nodes in the tree
  double validateTree() {
    if (root_ == nil_)
      return 0;
    else
      return std::sqrt(validateNode(root_));
  }

  /// fill @p depthMap with a list of nodes at each depth
  void generateDepthProfile(std::map<int, std::list<NodeRef> >& depthMap) {
    if (root_ == nil_) return;

    depthMap[0].push_back(root_);
    for (int i = 1; true; i++) {
      if (depthMap[i - 1].size() < 1) break;
      for (NodeRef node : depthMap[i - 1]) {
        if (left(node) != nil_) depthMap[i].push_back(left(node));
        if (right(node) != nil_) depthMap[i].push_back(right(node));
      }
    }
  }

  /// given @p @$ val \in [0,1] @$, sample a node from the weighted
  /// distribution
  template <typename T>
  NodeRef findInterval(T val) {
    if (root_ == nil_) return root_;

    return findInterval(root_, val * cum(root_));
  }
};

}  //< namespace btps
}  //< namespace mpblocks

#endif // TREE_H_
