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
 *  @file
 *
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef KD3_NODE_H_
#define KD3_NODE_H_

#include <Eigen/Dense>

namespace kd3 {

/// Base class for nodes in the kd tree
/**
 *  The Node data structure carries all the information that is actually
 *  required by the kd tree structure, including ancestry pointers and the
 *  point value for this node.
 *
 *  As a kind of design anomoly, the kd-tree employs the Curiously Recuring
 *  Template Pattern in the following way: the actual node type is expected
 *  to derive from Node<Traits>. Traits should contain a type Node which is
 *  the actual node type. Thus Node should be an inner class of Traits.
 *  Because kd_tree::Node<Traits> is a base class of the actual Node type we
 *  can cast it and use it as a kd_tree::Node<Traits> when working in the
 *  kd tree, but you can define the node structure to have all the extra
 *  information and methods you want.
 *
 *  The reason for this is that in applications the Node type really needs to
 *  have some sophisticated features. However, we need to ensure that it
 *  provides the proper interface and data storage to work inside the kd tree.
 *  It was easier to get things working if the Node knew its derived type,
 *  so that it can construct nodes and things.
 *
 *  No, I do not think this is good design... But it works.
 */
template <class Traits>
class Node {
 public:
  // these just shorten up some of the templated classes into smaller
  // names
  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::Node Derived;
  typedef typename Traits::HyperRect HyperRect;

  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> Vector;
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> Point;

  typedef Node<Traits> This;
//  typedef ListPair<Traits> Pair_t;

//  typedef NearestSearchIface<Traits> NNIface;
//  typedef RangeSearchIface<Traits> RangeIface;

 protected:
  unsigned int i_;          ///< dimension that this node splits on, also
                            ///  index of hyperplane's constant component
  Point point_;             ///< the point that this node contains
  Derived* parent_;         ///< parent node
  Derived* smaller_child_;  ///< child node who's i'th value is smaller
  Derived* greater_child_;  ///< child node who's i'th value is larger

 public:
  /// does nothing, see construct
  Node();

  /// construct a new node
  /**
   *  @param[in]  parent  the parent node of this node (0 for root node)
   *  @param[in]  i       the index of the dimensions on which this node
   *                      splits the space
   */
  void Construct(Node* parent, unsigned int i);

  /// fill point data (convenience method)
  void SetPoint(const Point& p) { point_ = p; }

  /// returns a Point_t of the point stored at this node
  const Point& GetPoint() { return point_; }

  /// return the parent node
  Derived* GetParent() { return parent_; }

  /// recursively inserts a new node in the tree, splitting the parent allong
  /// the longest dimension
  void Insert(HyperRect* hrect, Derived* node);

  /// perform a generic Nearest Neighbor query, different queries
  /// provide different implementations of the search structure
  /**
   *  @param q        query point for search
   *  @param rect     hyper rectangle containing this node
   *  @param search   implementation of search
   */
//  void FindNearest(const Point& q, HyperRect& rect, NNIface& search);

  /// find all nodes in the tree that lie inside the specified range
  /// ( can be arbitrary volume, which is implemented by the deriving
  /// class of the interface )
//  void FindRange(RangeIface& search, HyperRect& rect);

//  template <typename BackInserter>
//  void Enumerate(HyperRect& container, BackInserter bs);
};

}  // namespace kd3

#endif  // KD3_NODE_H_
