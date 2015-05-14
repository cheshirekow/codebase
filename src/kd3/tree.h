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
#include <kd3/enumerator.h>

namespace kd3 {

/// A simple kd-tree implementation
template <typename Scalar, int ndim_>
class Tree {
 public:
  /// a vector is the difference of two points
  typedef Eigen::Matrix<Scalar, ndim_, 1> Vector;

  /// the storage type for points
  typedef Eigen::Matrix<Scalar, ndim_, 1> Point;

 protected:
  Node<Scalar, ndim_>* root_;  ///< root node of the tree (0 if empty)
  int32_t size_;               ///< number of pointsSplit

  HyperRect<Scalar, ndim_> workspace_;  ///< total rectangle
  HyperRect<Scalar, ndim_> bounds_;     ///< bounding rectangle

 public:
  /// constructs a new kd-tree
  Tree() : root_(nullptr), size_(0) {}

  /// constructs a new kd-tree with a workspace
  Tree(const HyperRect<Scalar, ndim_>& workspace)
      : root_(nullptr), size_(0), workspace_(workspace) {}

  /// note that the tree does not own node data, and so will not free it
  ~Tree() {}

  /// Set the workspace, which is used as the initial hyper-rectangle for
  /// insertion queries
  void SetWorkspace(const HyperRect<Scalar, ndim_>& workspace) {
    workspace_ = workspace;
  }

  /// insert a node into the kd tree. The node should be newly created
  /// and contain no children
  /**
   *  @note   The tree does not take ownership of the node poitner
   */
  void Insert(Node<Scalar, ndim_>* node);

  /// generic NN search, specific search depends on the implementing
  /// class of the NNIface
  //  void findNearest(const Point_t& q, NNIface_t& search);

  /// generic range search, specific search depends on the implementing
  /// class of the RangeIface
  //  void findRange(RangeIface_t& search);

  std::list<typename Enumerator<Scalar, ndim_>::Pair> Enumerate() {
    return Enumerator<Scalar, ndim_>::BFS(root_, workspace_);
  }

  /// return the root node
  Node<Scalar, ndim_>* root() { return root_; }

  /// return the number of points in the tree
  int32_t size() { return size_; }

  /// clearout the data structure, note: does not destroy any object
  /// references
  void Clear() {
    root_ = 0;
    size_ = 0;
    bounds_ = HyperRect<Scalar, ndim_>();
  }
};

template <typename Scalar, int ndim_>
void Tree<Scalar, ndim_>::Insert(Node<Scalar, ndim_>* node) {
  size_++;
  const Point& pt = node->point();
  bounds_.GrowToContain(pt);

  HyperRect<Scalar, ndim_> node_rect = workspace_;
  if (root_) {
    root_->Insert(&node_rect, node);
  } else {
    root_ = node;
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
