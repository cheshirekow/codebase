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

#ifndef KD3_TREE_H_
#define KD3_TREE_H_

#include <cassert>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <kd3/node.h>
#include <kd3/hyperrect.h>

namespace kd3 {

/**
 *  @brief  a simple  KDtree class
 *  @tparam Traits    traits class for kd-tree specifying numer type, dimension
 *                    and the derived class for the node structure
 *  @see    kd_tree:Traits
 */
template <class Traits>
class Tree {
 public:
  /// number format, i.e. double, float
  typedef typename Traits::Scalar Scalar;

  /// the node class, should be defined as an inner class of Traits
  typedef typename Traits::Node Node;

  /// the hyper rectangle class shoudl be defined as an inner class of
  /// Traits, or a typedef in Traits
  typedef kd3::HyperRect<Traits> HyperRect;

  /// a vector is the difference of two points
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> Vector;

  /// the storage type for points
  typedef Vector Point;

  // these just shorten up some of the templated classes into smaller
  // names
  typedef Tree<Traits> This;
  typedef kd3::Node<Traits> NodeBase;
  //  typedef ListBuilder<Traits> ListBuilder_t;
  //  typedef NearestSearchIface<Traits> NNIface_t;
  //  typedef RangeSearchIface<Traits> RangeIface_t;

 protected:
  Node* root_;      ///< root node of the tree (0 if empty)
  HyperRect rect_;  ///< hyper rectangle for searches
  int32_t size_;        ///< number of points

  HyperRect workspace_;  ///< total rectangle
  HyperRect bounds_;     ///< bounding rectangle

 public:
  /// constructs a new kd-tree
  Tree() : root_(nullptr), size_(0) {}

  /// destructs the tree and recursively frees all node data.
  /// note that nodes do not own their data if their data are pointers
  ~Tree() {}

  /// insert a node into the kd tree. The node should be newly created
  /// and contain no children
  /**
   *  @param[in]  point   the k-dimensional point to insert into the
   *                      graph
   *  @param[in]  data    the data to store at the newly created node
   *                      (will most-likely contain *point)
   *
   *  @note   The tree does not take ownership of the node poitner
   */
  void Insert(Node* node);

  /// generic NN search, specific search depends on the implementing
  /// class of the NNIface
//  void findNearest(const Point_t& q, NNIface_t& search);

  /// generic range search, specific search depends on the implementing
  /// class of the RangeIface
//  void findRange(RangeIface_t& search);

  /// create a list of all the nodes in the tree, mostly only used for
  /// debug drawing
//  typename ListBuilder_t::List_t& buildList(bool bfs = true);

  /// return the list after buildList has been called, the reference is
  /// the same one returned by buildList but you may want to build the
  /// list and then use it multiple times later
//  typename ListBuilder_t::List_t& getList() { return m_lister.getList(); }

  /// return the root node
  Node* GetRoot() { return root_; }

  /// clearout the data structure, note: does not destroy any object
  /// references
  void Clear() {
    root_ = 0;
    size_ = 0;
    bounds_ = HyperRect();
  }

  /// return the number of points in the tree
  int32_t GetSize() { return size_; }
};

template <class Traits>
void Tree<Traits>::Insert(Node* node) {
  size_++;
  NodeBase* node_base = static_cast<NodeBase*>(node);
  const Point& pt = node_base->GetPoint();
  bounds_.GrowToContain(pt);

  HyperRect node_rect = workspace_;
  if (root_) {
    static_cast<NodeBase*>(root_)->Insert(node_rect, node);
  } else {
    root_ = node_;
    static_cast<NodeBase*>(root_)
        ->Construct(0, workspace_.GetLonestDimension());
  }
}


/*
template <class Traits>
void Tree<Traits>::findNearest(const Point_t& q, NNIface_t& search) {
  if (m_root) {
    m_rect = m_bounds;
    static_cast<NodeBase_t*>(m_root)->findNearest(q, m_rect, search);
  }
}

template <class Traits>
void Tree<Traits>::findRange(RangeIface_t& search) {
  if (m_root) {
    m_rect = m_bounds;
    static_cast<NodeBase_t*>(m_root)->findRange(search, m_rect);
  }
}

template <class Traits>
typename ListBuilder<Traits>::List_t& Tree<Traits>::buildList(bool bfs) {
  m_lister.reset();
  if (bfs)
    m_lister.buildBFS(m_root);
  else
    m_lister.buildDFS(m_root);

  return m_lister.getList();
}
*/


}  // namespace kd3

#endif  // KD3_TREE_H_
