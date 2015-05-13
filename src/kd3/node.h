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

#ifndef KD3_NODE_H_
#define KD3_NODE_H_

#include <functional>
#include <Eigen/Dense>
#include <kd3/hyperrect.h>

namespace kd3 {

/// Base class for nodes in the kd tree.
/**
 *  Derive from this class to associate extra data with points in the kd-tree
 *
 *  The Node data structure carries all the information that is actually
 *  required by the kd tree structure, including ancestry pointers and the
 *  point value for this node.
 */
template <typename Scalar, int ndim_>
class Node {
 public:
  typedef Eigen::Matrix<Scalar, ndim_, 1> Vector;
  typedef Eigen::Matrix<Scalar, ndim_, 1> Point;
  typedef kd3::HyperRect<Scalar, ndim_> HyperRect;
  typedef Node<Scalar, ndim_> ThisType;

  //  typedef ListPair<Traits> Pair_t;

  //  typedef NearestSearchIface<Traits> NNIface;
  //  typedef RangeSearchIface<Traits> RangeIface;

 protected:
  unsigned int i_;           ///< dimension that this node splits on, also
                             ///  index of hyperplane's constant component
  Point point_;              ///< the point that this node contains
  ThisType* smaller_child_;  ///< child node who's i'th value is smaller
  ThisType* greater_child_;  ///< child node who's i'th value is larger

 public:
  /// initializes fields to zero values
  Node() {
    point_.fill(0);
    i_ = 0;
    smaller_child_ = nullptr;
    greater_child_ = nullptr;
  }

  /// fill point data (convenience method)
  void SetPoint(const Point& p) { point_ = p; }

  /// returns the point stored at this node
  const Point& point() { return point_; }

  const ThisType* smaller_child() { return smaller_child_; }
  const ThisType* greater_child() { return greater_child_; }

  /// recursively inserts a new node in the tree, splitting the parent along
  /// the longest dimension
  /**
   * @param hrect hyper-rectangle covered by this node
   * @param node  node with the new point to insert
   * @todo  add SplitFn as a parameter defining the split rule
   */
  void Insert(HyperRect* hrect, ThisType* node);

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

template <typename Scalar, int ndim_>
void Node<Scalar, ndim_>::Insert(HyperRect* hrect, ThisType* node) {
  ThisType** ptr_child = 0;

  // first, grab a pointer to which child pointer we should recurse
  // into
  if (node->point_[i_] <= point_[i_]) {
    hrect->SplitLesser(i_, point_[i_]);
    ptr_child = &smaller_child_;
  } else {
    hrect->SplitGreater(i_, point_[i_]);
    ptr_child = &greater_child_;
  }

  ThisType*& child = *ptr_child;

  // if the child exists (is not null) then recurse, otherwise
  // create it and we're done
  if (child) {
    child->Insert(hrect, node);
  } else {
    child = node;
    node->i_ = hrect->GetLongestDimension();
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
