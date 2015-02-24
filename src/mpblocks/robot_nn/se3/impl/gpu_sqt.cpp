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

#ifndef MPBLOCKS_ROBOT_NN_SE3_GPU_SQT_H_
#define MPBLOCKS_ROBOT_NN_SE3_GPU_SQT_H_

#include <iostream>

#include <mpblocks/cudaNN/rect_dist.h>

#include "config.h"
#include "se3.h"
#include "quadtree.h"
#include "Implementation.h"
#include "impl/ctors.h"


namespace mpblocks {
namespace robot_nn {
namespace      se3 {


struct GPU_SQTImplementation:
    public Implementation
{
    std::vector<Point>& points;
    int                 capacity;
    int                 rate;

    cudaNN::so3_distance<false,float,4>  gpu_q_dist;
    cudaNN::RectangleQuery<float,4>           gpu_q_query;
    float                                     gpu_q_result[16];

    cudaNN::euclidean_distance<false,float,3>   gpu_x_dist;
    cudaNN::RectangleQuery<float,3>             gpu_x_query;
    float                                       gpu_x_result[16];

    quadtree::HyperRect<float,7> bounds;
    quadtree::Node               root;
    std::vector< quadtree::ScheduleItem> schedule;
    std::vector< quadtree::PriorityNode2<float,7> > queue;
    std::vector< quadtree::ResultNode<float> >      result;

    GPU_SQTImplementation( std::vector<Point>& points, int capacity, int rate ):
        points(points),
        capacity(capacity),
        rate(rate)
    {}

    virtual ~GPU_SQTImplementation(){}

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

        std::copy( q.data() + 0,
                   q.data() + 3, gpu_x_query.point );
        std::copy( q.data() + 3,
                   q.data() + 4, gpu_q_query.point );

        quadtree::HyperRect<float,7> cell = bounds;
        quadtree::pqueue_insert(queue,
                quadtree::PriorityNode2<float,7>(
                    0,0,
                    0,
                    &root,
                    cell),
                quadtree::PriorityNode2<float,7>::GreaterThan(w) );

        // while we have cells to process, we have not yet found k, and
        // the unprocessed cells contain at least one point which is closer
        // than the k'th so far, we must continue to expand cells
        while( queue.size() > 0 &&
                ( result.size() < k ||
                    queue[0].dist(w) < result[0].dist ) )
        {
            touched++;
            quadtree::PriorityNode2<float,7> pqnode;
            quadtree::pqueue_popmin(queue,pqnode,
                    quadtree::PriorityNode2<float,7>::GreaterThan(w));

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

                std::copy( pqnode.hrect[0].data() + 0,
                           pqnode.hrect[0].data() + 3, gpu_x_query.min );
                std::copy( pqnode.hrect[1].data() + 0,
                           pqnode.hrect[1].data() + 3, gpu_x_query.max );
                gpu_x_dist(gpu_x_query,gpu_x_result);

                std::copy( pqnode.hrect[0].data() + 3,
                           pqnode.hrect[0].data() + 7, gpu_q_query.min );
                std::copy( pqnode.hrect[1].data() + 3,
                           pqnode.hrect[1].data() + 7, gpu_q_query.max );
                gpu_q_dist(gpu_q_query,gpu_q_result);

//                // get the median of the cell
//                quadtree::Point<float,7> median =
//                        0.5f * ( pqnode.hrect.data[MIN] +
//                                 pqnode.hrect.data[MAX] );

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
                    float x_dist = pqnode.x_dist;
                    float q_dist = pqnode.q_dist;
                    if( sched->index == quadtree::INDEX_R3 )
                    {
                        int x_idx = (pair.first & 0x07) ;
                        x_dist = gpu_x_result[x_idx];
                    }
                    else
                    {
                        int q_idx = (pair.first & 0x78) >> 3;
                        q_dist = gpu_q_result[q_idx];
                    }

                    pqueue_insert( queue,
                        quadtree::PriorityNode2<float,7>(
                            x_dist,q_dist,
                            child_depth,
                            pair.second,
                            cell ),
                        quadtree::PriorityNode2<float,7>::GreaterThan(w) );
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


Implementation* impl_gpu_sqt(std::vector<Point>& points, int capacity, int rate )
{
    return new GPU_SQTImplementation(points,capacity,rate);
}

}
}
}



#endif // KD2_H_
