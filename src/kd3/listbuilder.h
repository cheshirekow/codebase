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
 *  @file   ListBuilder.h
 *
 *  @date   Feb 17, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef KD3_LISTBUILDER_H_
#define KD3_LISTBUILDER_H_

#include <deque>
#include <list>
#include <Eigen/Dense>

namespace kd3 {

/// pairs nodes of the Kd tree along with a hyperrectangle that is the
/// bounding volume for the subtree rooted at that node
template <class Traits>
struct ListPair {
  // these just shorten up some of the templated classes into smaller
  // names
  typedef typename Traits::Node Node;
  typedef typename Traits::HyperRect HyperRect;

  Node* node;
  HyperRect container;
};

/// Enumerates an entire subtree, building a list of nodes along with
/// the hyperectangle bounding the subtree at that node
/**
 *  @note lists can be built in breadth first or depth first order
 */
template <class Traits>
class ListBuilder {
 public:
  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::Node Node;
  typedef typename Traits::HyperRect HyperRect;

  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> Vector;
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> Point;

  // these just shorten up some of the templated classes into smaller
  // names
  typedef ListPair<Traits> Pair;
  typedef std::list<Pair> List;

  List DFS(Node* root) {
    ListBuilder builder;
    builder.BuildDFS(root);
    return builder.list_;
  }

  List BFS(Node* root) {
    ListBuilder builder;
    builder.BuildBFS(root);
    return builder.list_;
  }

 private:
  List deque_;
  List list_;

  /// build an enumeration of the tree
  /**
   *  @tparam Inserter_t  type of the insert iterator
   *  @param  root    root of the subtree to build
   *  @param  ins     the inserter where we put nodes we enumerate
   *                  should be an insertion iterator
   */
  template <typename Inserter>
  void Build(const HyperRect& hrect, Node* root, Inserter ins) {
    deque_.emplace_back({root, hrect});

    while (deque_.size() > 0) {
      list_.splice(list.end(), deque_, deque_.begin(), deque_.begin()++);
      const Pair& pair = list_.back();
      pair.node.Enumerate(pair.container, ins);
    }
  }

  /// enumerate a subtree in breadth-first manner
  void BuildBFS(Node* root) {
    Build(root, std::back_inserter(deque_));
  }

  /// enumerate a subtree in depth-first manner
  void BuildDFS(Node* root) {
    Build(root, std::front_inserter(deque_));
  }
};



}  // namespace kd3

#endif  // KD3_LISTBUILDER_H_
