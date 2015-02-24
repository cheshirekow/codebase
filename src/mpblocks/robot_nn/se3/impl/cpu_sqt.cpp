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

#include <iostream>
#include "config.h"
#include "Implementation.h"
#include "quadtree.h"
#include "impl/ctors.h"
#include "se3.h"

namespace mpblocks {
namespace robot_nn {
namespace      se3 {


struct SQTImplementation:
    public Implementation
{
    std::vector<Point>& points;
    int                 capacity;
    int                 rate;

    quadtree::HyperRect<float,7> bounds;
    quadtree::Node               root;
    std::vector< quadtree::ScheduleItem> schedule;
    std::vector< quadtree::PriorityNode<float,7> > queue;
    std::vector< quadtree::ResultNode<float> >     result;

//    KDImpl::Queue     queue;
//    KDImpl::ResultSet result;

    SQTImplementation( std::vector<Point>& points, int capacity, int rate ):
        points(points),
        capacity(capacity),
        rate(rate)
    {}

    virtual ~SQTImplementation(){}

    virtual void allocate( int maxSize )
    {
        quadtree::build_schedule(schedule,ws,w,rate,100);
        queue.reserve(maxSize*2);
        result.reserve(maxSize);

        for(int j=0; j < 3; j++)
        {
            bounds.data[robot_nn::MAX][j] = ws;
            bounds.data[robot_nn::MIN][j] = 0;
        }
        for(int j=3; j < 7; j++)
        {
            bounds.data[robot_nn::MIN][j] = -1.01;
            bounds.data[robot_nn::MAX][j] = 1.01;
        }
//        std::cout << "Quadtree schedule:\n";
//        for( auto item : schedule )
//            printf("   %4d,%d\n",item.depth,item.index);
    }

    virtual void insert_derived( int i, const Point& x )
    {
        quadtree::HyperRect<float,7> cell = bounds;
        quadtree::scheduled_insert<float>(
            points,i,cell,&root,capacity,&schedule[0] );
    }

    virtual void findNearest_derived( const Point& q )
    {
        touched=0;
        queue.clear();
        result.clear();

        quadtree::HyperRect<float,7> cell = bounds;
        quadtree::pqueue_insert(queue,
                quadtree::PriorityNode<float,7>(
                    0,
                    0,
                    &root,
                    cell) );

        // while we have cells to process, we have not yet found k, and
        // the unprocessed cells contain at least one point which is closer
        // than the k'th so far, we must continue to expand cells
        while( queue.size() > 0 &&
                ( result.size() < k ||
                    queue[0].dist < result[0].dist ) )
        {
            touched++;
            quadtree::PriorityNode<float,7> pqnode;
            quadtree::pqueue_popmin(queue,pqnode);

//            std::cout << "popped cell:"
//                      << "\n   " <<  pqnode.hrect.data[MIN].transpose()
//                      << "\n   " <<  pqnode.hrect.data[MAX].transpose()
//                      << "\n";

            quadtree::Node* node = pqnode.node;
            // if the node is a leaf node, process all of it's entries
            if( node->type == quadtree::LEAF )
            {
                for(int pointRef : node->points )
                {
                    quadtree::result_insert( result,
                        quadtree::ResultNode<float>(
                            se3::se3_distance(w,q,points[pointRef])
                            ,pointRef) );
                    if( result.size() > k )
                        quadtree::result_popmax(result);
                }
            }
            // otherwise queue up it's children
            else
            {
                // get the depth of the children
                int child_depth = pqnode.depth+1;

                // get the median of the cell
                quadtree::Point<float,7> median =
                        0.5f * ( pqnode.hrect.data[MIN] +
                                 pqnode.hrect.data[MAX] );

                // get the schedule for that depth
                std::vector< quadtree::ScheduleItem>::iterator sched =
                std::lower_bound(
                        schedule.begin(),
                        schedule.end(),
                        quadtree::ScheduleItem(child_depth,0),
                        quadtree::ScheduleItem::LessThan() );
                if( sched == schedule.end() )
                    std::cerr << "no schedule for depth " << child_depth;

                for( auto& pair : node->children )
                {
                    float dist;
                    bool  feas;

//                    if( pair.first > (0x01 << 3) )
//                        std::cerr << "large child index: "
//                                  << pair.first << ", depth "
//                                  << child_depth << ", sched "
//                                  << sched->index << "\n";
//                    int x_idx = (pair.first & 0x07) ;
//                    int q_idx = (pair.first & 0x78) >> 3;

                    quadtree::HyperRect<float,7> cell = pqnode.hrect;
                    if( sched->index == quadtree::INDEX_R3 )
                    {
                        std::bitset<3> bits( pair.first );
                        quadtree::refine<float,0,3,7>(
                                cell,median,bits );
                    }
                    else
                    {
                        std::bitset<4> bits( pair.first );
                        quadtree::refine<float,3,4,7>(
                                cell,median,bits );
                    }

                    se3::se3_distance<float>(w,q,cell,dist,feas);
                    if(!feas)
                        std::cerr << "infeasible cell with children\n";

                    pqueue_insert( queue,
                        quadtree::PriorityNode<float,7>(
                            dist,
                            child_depth,
                            pair.second,
                            cell ) );
                }
            }
        }
    }

    virtual void get_result( std::vector<int>& out )
    {
        out.clear();
        out.reserve(k);
        for(auto& item : result )
            out.push_back( item.point );
    }
};


Implementation* impl_cpu_sqt(std::vector<Point>& points, int capacity, int rate )
{
    return new SQTImplementation(points,capacity,rate);
}

}
}
}



#endif // KD2_H_
