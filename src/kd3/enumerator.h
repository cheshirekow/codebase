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

#ifndef KD3_ENUMERATOR_H_
#define KD3_ENUMERATOR_H_

#include <list>
#include <tuple>
#include <Eigen/Dense>

namespace kd3 {

/// Enumerates an entire subtree, building a list of nodes along with
/// the hyperectangle bounding the subtree at that node
/**
 *  @TODO make a generator version for range based loops with no storage
 *        requirements
 */
template <typename Scalar, int ndim_>
class Enumerator {
 public:
  typedef kd3::Node<Scalar, ndim_> Node;
  typedef kd3::HyperRect<Scalar, ndim_> HyperRect;
  typedef Eigen::Matrix<Scalar, ndim_, 1> Vector;
  typedef Eigen::Matrix<Scalar, ndim_, 1> Point;

  /// pairs nodes of the Kd tree along with a hyperrectangle that is the
  /// bounding volume for the subtree rooted atalong that node
  struct Pair {
    const Node* node;
    HyperRect container;

    Pair(const Node* node, const HyperRect& container)
        : node(node), container(container) {}
  };

  static std::list<Pair> BFS(const Node* root, const HyperRect& hrect) {
    Enumerator builder;
    builder.Build(root, hrect);
    return builder.list_;
  }

 private:
  std::list<Pair> deque_;
  std::list<Pair> list_;

  /// build an enumeration of the tree
  /**
   *  @tparam Inserter  type of the insert iterator
   *  @param  root    root of the subtree to build
   *  @param  ins     the inserter where we put nodes we enumerate
   *                  should be an insertion iterator
   */
  void Build(const Node* root, const HyperRect& hrect) {
    deque_.emplace_back(root, hrect);

    while (deque_.size() > 0) {
      auto begin_iter = deque_.begin();
      auto end_iter = deque_.end();
      ++end_iter;
      list_.splice(list_.end(), deque_, begin_iter, end_iter);
      const Pair& pair = list_.back();
      HyperRect left_container, right_container;
      pair.node->Split(pair.container, &left_container, &right_container);
      deque_.emplace_back(pair.node->smaller_child(), left_container);
      deque_.emplace_back(pair.node->greater_child(), right_container);
    }
  }

};

}  // namespace kd3

#endif  // KD3_ENUMERATOR_H_
