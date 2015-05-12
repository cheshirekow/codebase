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
  typedef Node<Traits> Base;

  //  typedef ListPair<Traits> Pair_t;

  //  typedef NearestSearchIface<Traits> NNIface;
  //  typedef RangeSearchIface<Traits> RangeIface;

 protected:
  unsigned int i_;          ///< dimension that this node splits on, also
                            ///  index of hyperplane's constant component
  Point point_;             ///< the point that this node contains
  Derived* smaller_child_;  ///< child node who's i'th value is smaller
  Derived* greater_child_;  ///< child node who's i'th value is larger

 public:
  /// does nothing, see construct
  Node() {
    parent_ = nullptr;
    i_ = 0;
    smaller_child_ = nullptr;
    greater_child_ = nullptr;
  }

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

template <class Traits>
void Node<Traits>::Insert(HyperRect* hrect, Derived* node) {
  Derived** ptr_child = 0;

  // first, grab a pointer to which child pointer we should recurse
  // into
  if (static_cast<Base*>(node)->point_[i_] <= point_[i_]) {
    hrect->SplitLesser(i_, point_[i_]);
    ptr_child = &smaller_child;
  } else {
    hrect->SplitGreater(i_, point_[i_]);
    ptr_child = &greater_child;
  }

  Derived*& child = *ptr_child;

  // if the child exists (is not null) then recurse, otherwise
  // create it and we're done
  if (child)
    static_cast<This*>(child)->Insert(node);
  else {
    child = node;
    static_cast<This*>(node)->i_ = hrect->GetLongestDimension();
  }
}

/*
template <class Traits>
void Node<Traits>::findNearest(const Point_t& q, HyperRect_t& rect,
                               NNIface_t& search) {
  Node_t* nearerNode;
  Node_t* fartherNode;
  Format_t* nearerHyperCoord;
  Format_t* fartherHyperCoord;

  // first, check to see if the left child or the right child is a
  // would-be-ancester if the point were incerted in the graph
  Format_t diff = q[m_i] - m_point[m_i];
  if (diff <= 0.0) {
    nearerNode = m_smallerChild;
    fartherNode = m_greaterChild;

    nearerHyperCoord = &(rect.maxExt[m_i]);
    fartherHyperCoord = &(rect.minExt[m_i]);
  }

  else {
    nearerNode = m_greaterChild;
    fartherNode = m_smallerChild;

    nearerHyperCoord = &(rect.minExt[m_i]);
    fartherHyperCoord = &(rect.maxExt[m_i]);
  }

  // now, whichever child is the would-be-ancestor, recurse into them
  // also, refine the hyperrectangle that contains the node by
  // splitting it along the split-plane of this node
  if (nearerNode) {
    // copy out the old extent of they hyper-rectangle
    Format_t oldHyperVal = *nearerHyperCoord;

    // split the hyperrectangle by updating the extent with the
    // value of this nodes split plane
    *nearerHyperCoord = m_point[m_i];

    // recurse into the nearer node
    static_cast<This_t*>(nearerNode)->findNearest(q, rect, search);

    // now that we've stepped back up into this node, restore the
    // hyperrectangle
    *nearerHyperCoord = oldHyperVal;
  }

  // evaluate this node and add it to the result set if necessary
  search.evaluate(q, m_point, m_this);

  // if the farther node exists, we might need to also check it's
  // children
  if (fartherNode) {
    // refine the hyper-rectangle of the farther subtree
    Format_t oldHyperVal = *fartherHyperCoord;
    *fartherHyperCoord = m_point[m_i];

    // check if we have to recurse into the farther subtree
    if (search.shouldRecurse(q, rect))
      static_cast<This_t*>(fartherNode)->findNearest(q, rect, search);

    // undo refinement of the hyperrectangle
    *fartherHyperCoord = oldHyperVal;
  }
}

template <class Traits>
void Node<Traits>::findRange(RangeIface_t& search, HyperRect_t& rect) {
  // evaluate this node and add it to the result set if
  // it lies inside the range
  search.evaluate(m_point, m_this);

  // now evaluate the two children and recurse if necessary
  {
    // copy out the old extent of they hyper-rectangle
    Node_t* child = m_smallerChild;
    Format_t* hyperCoord = &(rect.maxExt[m_i]);
    Format_t oldHyperVal = *hyperCoord;

    // split the hyperrectangle by updating the extent with the
    // value of this nodes split plane
    *hyperCoord = m_point[m_i];

    // recurse into the nearer node
    if (child && search.shouldRecurse(rect))
      static_cast<This_t*>(child)->findRange(search, rect);

    // now that we've stepped back up into this node, restore the
    // hyperrectangle
    *hyperCoord = oldHyperVal;
  }

  {
    // copy out the old extent of they hyper-rectangle
    Node_t* child = m_greaterChild;
    Format_t* hyperCoord = &(rect.minExt[m_i]);
    Format_t oldHyperVal = *hyperCoord;

    // split the hyperrectangle by updating the extent with the
    // value of this nodes split plane
    *hyperCoord = m_point[m_i];

    // recurse into the nearer node
    if (child && search.shouldRecurse(rect))
      static_cast<This_t*>(child)->findRange(search, rect);

    // now that we've stepped back up into this node, restore the
    // hyperrectangle
    *hyperCoord = oldHyperVal;
  }
}

template <class Traits>
template <typename BackInserter>
void Node<Traits>::enumerate(HyperRect_t& container, BackInserter bs) {
  if (m_greaterChild) {
    Pair_t* pair = new Pair_t();
    container.copyTo(pair->container);
    pair->container.minExt[m_i] = m_point[m_i];
    pair->node = m_greaterChild;
    bs = pair;
  }

  if (m_smallerChild) {
    Pair_t* pair = new Pair_t();
    container.copyTo(pair->container);
    pair->container.maxExt[m_i] = m_point[m_i];
    pair->node = m_smallerChild;
    bs = pair;
  }
}
*/

}  // namespace kd3

#endif  // KD3_NODE_H_
