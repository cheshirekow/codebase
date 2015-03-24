/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of mpblocks.
 *
 *  mpblocks is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  mpblocks is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with mpblocks.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  \file   /mpblocks/kd_tree.h
 *
 *  \date   Sep 9, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef MPBLOCKS_KD_TREE_H_
#define MPBLOCKS_KD_TREE_H_

#include <Eigen/Dense>

namespace         mpblocks {

/// implements a kd-tree, a multidimensional search tree for points
/**
 *  A kd-tree is a binary space partition where each point in the tree
 *  also defines a split plane. Each split plane is axis aligned, and if
 *  a node at depth @f$ d @f$ splits along the @f$ i @f$'th dimension, then the
 *  nodes at depth @f$ d + 1 @f$ split along the @f$ (i+1) \% k @f$ dimension.
 */
namespace          kd_tree {

// forward declarations
template <class Traits>     struct ListPair;
template <class Traits>     struct ListBuilder;
template <class Traits>     class  Node;
template <class Traits>     class  Tree;
template <class Traits>     class  NearestSearchIface;
template <class Traits>     class  RangeSearchIface;


} // namespace kd_tree
} // mpblocks


#include <mpblocks/kd_tree/ListBuilder.h>
#include <mpblocks/kd_tree/ListPair.h>
#include <mpblocks/kd_tree/NearestSearch.h>
#include <mpblocks/kd_tree/Node.h>
#include <mpblocks/kd_tree/RangeSearch.h>
#include <mpblocks/kd_tree/Tree.h>

#include <mpblocks/kd_tree/euclidean.h>
#include <mpblocks/kd_tree/r2_s1.h>










#endif // KD_TREE_H_
