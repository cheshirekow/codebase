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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/r2s1/impl/kd2.h
 *
 *  @date   Nov 7, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_KD2_H_
#define MPBLOCKS_KD2_H_

#include "config.h"
#include "Implementation.h"
#include "insert.h"
#include "findNearest.h"
#include "functions.h"
#include "impl/ctors.h"

namespace mpblocks {
namespace robot_nn {



struct SKDImplementation:
    public Implementation
{
    typedef KDStructs<float,3> KDImpl;
    std::vector<Point>&         points;
    std::vector<KDImpl::Node>   nodes;
    KDImpl::Queue     queue;
    KDImpl::ResultSet result;
    KDImpl::HyperRect bounds;

    SKDImplementation( std::vector<Point>& points ):
        points(points)
    {}

    virtual ~SKDImplementation(){}

    virtual void allocate( int maxSize )
    {
        nodes.reserve(maxSize*2);
        nodes.push_back( KDImpl::Node() );
    }

    virtual void insert_derived( int i, const Point& x )
    {
        for(int j=0; j < 3; j++)
        {
            if( x[j] > bounds.data[robot_nn::MAX][j] )
                bounds.data[robot_nn::MAX][j] = x[j];
            if( x[j] < bounds.data[robot_nn::MIN][j] )
                bounds.data[robot_nn::MIN][j] = x[j];
        }

        KDImpl::HyperRect cell = bounds;
        robot_nn::insert( cell, &nodes[0], i,
                    KDImpl::IsLeafFn(),
                    KDImpl::ScheduledLeafInsertFn(nodes,points,w),
                    KDImpl::IdxFn(),
                    KDImpl::ValueFn(),
                    KDImpl::ChildFn(),
                    KDImpl::PointGetFn(points),
                    KDImpl::HyperSetFn() );
    }

    virtual void findNearest_derived( const Point& q )
    {
        queue.clear();
        result.clear();
        touched = 0;
        queue.push_back( KDImpl::PriorityNode(0,&nodes[0],bounds) );
        robot_nn::findNearest(q,k,queue,result,
                    KDImpl::QueueMinKeyFn(),
                    KDImpl::QueuePopMinFn(),
                    KDImpl::QueueInsertFn(touched),
                    KDImpl::QueueSizeFn(),
                    KDImpl::SetMaxKeyFn(),
                    KDImpl::SetPopMaxFn(),
                    KDImpl::SetInsertFn(),
                    KDImpl::SetSizeFn(),
                    KDImpl::R2S1PointDistFn(points,w),
                    KDImpl::R2S1CellDistFn(w),
//                    KDImpl::PointDistFn(points),
//                    KDImpl::CellDistFn(),
                    KDImpl::IsLeafFn(),
                    KDImpl::ChildrenFn(),
                    KDImpl::SitesFn(),
                    KDImpl::CellFn(),
                    KDImpl::NodeFn() );
    }

    virtual void get_result( std::vector<int>& out )
    {
        out.clear();
        out.reserve(k);
        for(auto& item : result )
            out.push_back( item.point );
    }
};

Implementation* impl_skd(std::vector<Point>& points )
{
    return new SKDImplementation(points);
}


}
}



#endif // KD2_H_
