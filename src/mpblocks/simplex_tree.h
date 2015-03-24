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
 *  @file   mpblocks/simplex_tree/simplex_tree.h
 *
 *  @date   Dec 2, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_SIMPLEX_TREE_H_
#define MPBLOCKS_SIMPLEX_TREE_H_


namespace mpblocks     {

/// classes implemeting a simplex tree for proximity queries
/**
 *  A simplex tree is like a quad tree. It's a recursive decomposition of
 *  a point set into those that lie inside a simplex. Each simplex is
 *  subdivided into NDim other simplices. The location of the split depends
 *  on the split policy
 */
namespace simplex_tree {

template <class Traits> class Geometry;
template <class Traits> class Node;
template <class Traits> class Simplex;
template <class Traits> class Tree;

}   // namespace simplex_tree
}   // namespace mpblocks

#include <mpblocks/simplex_tree/Geometry.h>
#include <mpblocks/simplex_tree/Node.h>
#include <mpblocks/simplex_tree/Simplex.h>
#include <mpblocks/simplex_tree/Tree.h>




#endif // SIMPLEX_TREE_H_
