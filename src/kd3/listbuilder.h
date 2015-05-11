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
  typedef ListPair<Traits> Pait;
  typedef std::deque<Pair*> Deque;
  typedef std::list<Pair*> List;

 private:
  Deque deque_;
  List list_;
  HyperRect hyper_;

 public:
  /// build an enumeration of the tree
  /**
   *  @tparam Inserter_t  type of the insert iterator
   *  @param  root    root of the subtree to build
   *  @param  ins     the inserter where we put nodes we enumerate
   *                  should be an insertion iterator
   */
  template <typename Inserter>
  void Build(Node* root, Inserter ins);

  /// enumerate a subtree in breadth-first manner
  void BuildBFS(Node* root);

  /// enumerate a subtree in depth-first manner
  void BuildDFS(Node* root);

  /// return the list
  List& GetList();
};

template <class Traits>
void ListBuilder<Traits>::reset() {
  for (typename Deque_t::iterator ipPair = m_deque.begin();
       ipPair != m_deque.end(); ipPair++)
    delete *ipPair;

  for (typename List_t::iterator ipPair = m_list.begin();
       ipPair != m_list.end(); ipPair++)
    delete *ipPair;

  m_list.clear();
  m_deque.clear();
  m_hyper.makeInfinite();
}

template <class Traits>
template <typename Inserter_t>
void ListBuilder<Traits>::build(Node_t* root, Inserter_t ins) {
  Pair_t* rootPair = new Pair_t();
  rootPair->node = root;
  m_hyper.copyTo(rootPair->container);
  m_deque.push_back(rootPair);

  while (m_deque.size() > 0) {
    Pair_t* pair = m_deque.front();
    m_deque.pop_front();
    m_list.push_back(pair);
    pair->node->enumerate(pair->container, ins);
  }
}

template <class Traits>
void ListBuilder<Traits>::BuildBFS(Node* root) {
  Build(root, std::back_inserter(deque_));
}

template <class Traits>
void ListBuilder<Traits>::BuildDFS(Node* root) {
  Build(root, std::front_inserter(deque_));
}

}  // namespace kd3

#endif // KD3_LISTBUILDER_H_
