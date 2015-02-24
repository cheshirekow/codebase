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
 *  @file   /home/josh/Codes/cpp/mpblocks2/kdtree/include/mpblocks/kd2/insert.h
 *
 *  @date   Nov 3, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_ROBOT_NN_INSERT_H_
#define MPBLOCKS_ROBOT_NN_INSERT_H_

#include <iostream>
#include "BinaryKey.h"

namespace mpblocks {
namespace robot_nn {


template<
    class NodeRef,
    class PointRef,
    class IsLeafFn,
    class LeafInsertFn,
    class IdxFn,
    class ValueFn,
    class ChildFn,
    class PointGetFn
>
void insert(NodeRef         node,           ///< the current node
            PointRef        point,          ///< the point to be inserted
            const IsLeafFn&       isLeaf,         ///< returns true if node is a leaf
            const LeafInsertFn&   leafInsert,     ///< inserts point into a leaf node's store
            const IdxFn&          idx,            ///< returns the split index of a node
            const ValueFn&        value,          ///< returns the split value of a node
            const ChildFn&        child,          ///< returns the children of a node
            const PointGetFn&     pointGet        ///< returns the i'th value of a point
            )
{
    NodeRef parent = node;
    // walk down the tree until we reach  a leaf
    while( !isLeaf( node ) )
    {
        int       i      = idx( node );
        BinaryKey which  = ( pointGet(point,i) <= value(node) ) ? MIN : MAX;
        node = child(node,which);
    }

    // if we've reached a leaf node, then insert the point, if the leaf needs
    // to split then this function also does the splitting
    leafInsert(node,point);
}


template<
    class HyperRect,
    class NodeRef,
    class PointRef,
    class IsLeafFn,
    class LeafInsertFn,
    class IdxFn,
    class ValueFn,
    class ChildFn,
    class PointGetFn,
    class HyperSetFn
>
void insert(HyperRect&      cell,           ///< hyper rectangle of node
            NodeRef         node,           ///< the current node
            PointRef        point,          ///< the point to be inserted
            const IsLeafFn&       isLeaf,         ///< returns true if node is a leaf
            const LeafInsertFn&   leafInsert,     ///< inserts point into a leaf node's store
            const IdxFn&          idx,            ///< returns the split index of a node
            const ValueFn&        value,          ///< returns the split value of a node
            const ChildFn&        child,          ///< returns the children of a node
            const PointGetFn&     pointGet,       ///< returns the i'th value of a point
            const HyperSetFn&     hyperSet
            )
{
    NodeRef parent = node;

    // walk down the tree until we reach  a leaf
    while( !isLeaf( node ) )
    {
        int       i      = idx( node );
        BinaryKey which  = ( pointGet(point,i) <= value(node) ) ? MIN : MAX;
        node = child(node,which);
        hyperSet(cell,other(which),i,value(node));
    }

    // if we've reached a leaf node, then insert the point, if the leaf needs
    // to split then this function also does the splitting
    leafInsert(cell,node,point);
}








} //< namespace kd2
} //< namespace mpblocks















#endif // INSERT_H_
