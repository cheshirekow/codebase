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

#ifndef MPBLOCKS_KD_TREE_NODE_H_
#define MPBLOCKS_KD_TREE_NODE_H_

namespace mpblocks {
namespace kd_tree {

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
  typedef typename Traits::Format_t Format_t;
  typedef typename Traits::Node Node_t;
  typedef typename Traits::HyperRect HyperRect_t;

  typedef Eigen::Matrix<Format_t, Traits::NDim, 1> Vector_t;
  typedef Vector_t Point_t;

  typedef Node<Traits> This_t;
  typedef ListPair<Traits> Pair_t;

  typedef NearestSearchIface<Traits> NNIface_t;
  typedef RangeSearchIface<Traits> RangeIface_t;

 protected:
  unsigned int m_i;        ///< dimension that this node splits on, also
                           ///  index of hyperplane's constant component
  Point_t m_point;         ///< the point that this node contains
  Node_t* m_parent;        ///< parent node
  Node_t* m_smallerChild;  ///< child node who's i'th value is smaller
  Node_t* m_greaterChild;  ///< child node who's i'th value is larger
  Node_t* m_this;

 public:
  /// does nothing, see construct
  Node();

  /// construct a new node
  /**
   *  @param[in]  parent  the parent node of this node (0 for root node)
   *  @param[in]  i       the index of the dimensions on which this node
   *                      splits the space
   */
  void construct(Node_t* parent, unsigned int i);

  /// fill point data (convenience method)
  void setPoint(const Point_t& p);

  /// returns a Point_t of the point stored at this node
  const Point_t& getPoint();

  /// return the parent node
  Node_t* getParent() { return m_parent; }

  /// recursively inserts point as a new node in the tree
  void insert(Node_t*);

  /// perform a generic Nearest Neighbor query, different queries
  /// provide different implementations of the search structure
  /**
   *  @param q        query point for search
   *  @param rect     hyper rectangle containing this node
   *  @param search   implementation of search
   */
  void findNearest(const Point_t& q, HyperRect_t& rect, NNIface_t& search);

  /// find all nodes in the tree that lie inside the specified range
  /// ( can be arbitrary volume, which is implemented by the deriving
  /// class of the interface )
  void findRange(RangeIface_t& search, HyperRect_t& rect);

  template <typename BackInserter>
  void enumerate(HyperRect_t& container, BackInserter bs);
};

}  // namespace kd_tree
}  // mpblocks

#endif
