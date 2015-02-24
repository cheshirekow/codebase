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
#include "quadtree.h"
#include "impl/ctors.h"

namespace mpblocks {
namespace robot_nn {

namespace quadtree {

void build_schedule( std::vector<ScheduleItem>& schedule,
                     float s0, float w, int rate, int depth )
{
    double frac = s0/(2*M_PI*w);
    int    exp;
    std::frexp(frac,&exp);
    int    log2frac = exp-1;

    schedule.clear();
    schedule.reserve(depth);

    for(int i=0; i*rate + log2frac < depth; i++)
    {
        int idx = (i%2 == 0) ? INDEX_R2 : INDEX_R1;
        schedule.push_back( ScheduleItem(i*rate + log2frac, idx) );
    }
}

}

struct SQTImplementation:
    public Implementation
{
    std::vector<Point>& points;
    int                 capacity;
    int                 rate;

    quadtree::HyperRect<float,3> bounds;
    quadtree::Node               root;
    std::vector< quadtree::ScheduleItem> schedule;
    std::vector< quadtree::PriorityNode<float,3> > queue;
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

        for(int j=0; j < 2; j++)
        {
            bounds.data[robot_nn::MAX][j] = ws;
            bounds.data[robot_nn::MIN][j] = 0;
        }
        bounds.data[robot_nn::MIN][2] = -M_PI;
        bounds.data[robot_nn::MAX][2] = M_PI;
//        std::cout << "Quadtree schedule:\n";
//        for( auto item : schedule )
//            printf("   %4d,%d\n",item.depth,item.index);
    }

    virtual void insert_derived( int i, const Point& x )
    {
        quadtree::HyperRect<float,3> cell = bounds;
        quadtree::scheduled_insert<float>(
            points,i,cell,&root,capacity,&schedule[0] );
    }

    virtual void findNearest_derived( const Point& q )
    {
        touched=0;
        queue.clear();
        result.clear();

        quadtree::HyperRect<float,3> cell = bounds;
        quadtree::pqueue_insert(queue,
                quadtree::PriorityNode<float,3>(
                    quadtree::r2s1_distance<float,3>(q,cell,w),
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
            quadtree::PriorityNode<float,3> pqnode;
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
                            quadtree::r2s1_distance(q,points[pointRef],w)
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

                // get the schedule for that depth
                std::vector< quadtree::ScheduleItem>::iterator sched =
                std::lower_bound(
                        schedule.begin(),
                        schedule.end(),
                        quadtree::ScheduleItem(child_depth,0),
                        quadtree::ScheduleItem::LessThan() );

                // get the median of the cell
                quadtree::Point<float,3> median =
                        0.5f * ( pqnode.hrect.data[MIN] +
                                 pqnode.hrect.data[MAX] );

                for( auto& pair : node->children )
                {
                    quadtree::HyperRect<float,3> cell = pqnode.hrect;
                    if( sched->index == quadtree::INDEX_R2 )
                    {
                        std::bitset<2> bits( pair.first );
                        quadtree::refine<float,0,2,3>(
                                cell,median,bits );
                        pqueue_insert( queue,
                            quadtree::PriorityNode<float,3>(
                                quadtree::r2s1_distance<float,3>(q,cell,w),
                                child_depth,
                                pair.second,
                                cell ) );
                    }
                    else
                    {
                        std::bitset<1> bits( pair.first );
                        quadtree::refine<float,2,1,3>(
                                cell,median,bits );
                        pqueue_insert( queue,
                            quadtree::PriorityNode<float,3>(
                                quadtree::r2s1_distance<float,3>(q,cell,w),
                                child_depth,
                                pair.second,
                                cell ) );
                    }
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


Implementation* impl_sqt(std::vector<Point>& points, int capacity, int rate )
{
    return new SQTImplementation(points,capacity,rate);
}


}
}



#endif // KD2_H_
