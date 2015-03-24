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
 *  @file   mpblocks/simplex_tree/Node.h
 *
 *  @date   Dec 2, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_SIMPLEX_TREE_NODE_H_
#define MPBLOCKS_SIMPLEX_TREE_NODE_H_

#include <Eigen/Dense>
#include <vector>
#include <list>

namespace mpblocks     {
namespace simplex_tree {

/// base class for nodes, implements storage and interface for the simplex tree
/**
 *  A node is something which is inserted into the tree. It must provide a
 *  notion of a point in @f \mathbb{R}^n @f.
 *
 *  The proper usage is to define a member class of Traits which is the
 *  actual node class, and which derives from this class. By CTRP it will
 *  contain storage and the necessary interface for the simplex tree, but the
 *  derived class is always returned through API calls
 */
template <class Traits>
class Node
{
    public:
        typedef typename Traits::Format         Format_t;
        typedef typename Traits::Node           Node_t;

        typedef unsigned int                                        Index_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim,1>              Vector_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim,1>              Point_t;

    protected:
        Point_t m_point;

    public:
        const Point_t& getPoint();

};






}   // namespace simplex_tree
}   // namespace mpblocks












#endif // NODE_H_
