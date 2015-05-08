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
 *  @file   ListPair.h
 *
 *  @date   Feb 17, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_LISTPAIR_H_
#define MPBLOCKS_KD_TREE_LISTPAIR_H_

namespace mpblocks {
namespace kd_tree {

/// pairs nodes of the Kd tree along with a hyperrectangle that is the
/// bounding volume for the subtree rooted at that node
template <class Traits>
struct ListPair {
  // these just shorten up some of the templated classes into smaller
  // names
  typedef typename Traits::Node Node_t;
  typedef typename Traits::HyperRect HyperRect_t;

  Node_t* node;
  HyperRect_t container;
};

}  // namespace kd_tree
}  // mpblocks

#endif /* LISTPAIR_H_ */
