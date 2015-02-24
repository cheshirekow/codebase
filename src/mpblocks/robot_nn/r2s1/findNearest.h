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
 *  @file   include/mpblocks/kd2/findNearest.h
 *
 *  @date   Nov 3, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_ROBOT_NN_FINDNEAREST_H_
#define MPBLOCKS_ROBOT_NN_FINDNEAREST_H_

#include "BinaryKey.h"

namespace mpblocks {
namespace robot_nn {

template <
    // implicit types
    class Point,
    // input types
    class SearchQueue,
    class ResultSet,
    // queue operations
    class QueueMinKeyFn,
    class QueuePopMinFn,
    class QueueInsertFn,
    class QueueSizeFn,
    // set operations
    class SetMaxKeyFn,
    class SetPopMaxFn,
    class SetInsertFn,
    class SetSizeFn,
    // distance functions
    class PointDistanceFn,
    class CellDistanceFn,
    // node functions
    class IsLeafFn,
    class ChildrenFn,
    class SitesFn,
    // pair functions
    class CellFn,
    class NodeFn
>
void findNearest( const Point& query,
                  int k,
                  SearchQueue& q,
                  ResultSet&   r,
                  const QueueMinKeyFn& minKey,
                  const QueuePopMinFn& popMin,
                  const QueueInsertFn& enqueue,
                  const QueueSizeFn&   queueSize,
                  const SetMaxKeyFn&   maxKey,
                  const SetPopMaxFn&   popMax,
                  const SetInsertFn&   insert,
                  const SetSizeFn&     size,
                  const PointDistanceFn& pointDist,
                  const CellDistanceFn&  cellDist,
                  const IsLeafFn&        isLeaf,
                  const ChildrenFn&      children,
                  const SitesFn&         sites,
                  const CellFn&          getCell,
                  const NodeFn&          getNode)
{
    // if the nearest point in the unsearched frontier is further away than
    // the k-th nearest point found then we have found the k-NN
    while( queueSize(q) > 0 && (size(r) < k || minKey(q) < maxKey(r)) )
    {
        // at each iteration expand the unsearched cell which contains the
        // nearest point to the query
        auto pair = popMin(q);
        auto node = getNode(pair);

        // if it's a leaf node then evaluate the distance to each site
        // contained in that nodes cell
        if( isLeaf( node ) )
        {
            for( auto site : sites(node) )
            {
                insert( r, pointDist(query,site), site );
                if( size(r) > k )
                    popMax(r);
            }
        }
        else
        {
            for( auto childPair : children(pair) )
                enqueue(q,cellDist(query,getCell(childPair)),childPair);
        }
    }
}



} //< namespace kd2
} //< namespace mpblocks















#endif // FINDNEAREST_H_
