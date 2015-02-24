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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/so3.h
 *
 *  @date   Nov 12, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_ROBOT_NN_SE3_RQT_H_
#define MPBLOCKS_ROBOT_NN_SE3_RQT_H_

#include <cmath>
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

/// regular quad trees
struct GPU_RQTImplementation:
    public Implementation
{
    std::vector<Point>& points;
    int                 capacity;

    cudaNN::so3_distance<false,float,4>  gpu_q_dist;
    cudaNN::RectangleQuery<float,4>           gpu_q_query;
    float                                     gpu_q_result[16];

    cudaNN::euclidean_distance<false,float,3>   gpu_x_dist;
    cudaNN::RectangleQuery<float,3>             gpu_x_query;
    float                                       gpu_x_result[16];


    quadtree::HyperRect<float,7> bounds;
    quadtree::Node               root;
    std::vector< quadtree::PriorityNode<float,7> > queue;
    std::vector< quadtree::ResultNode<float> >     result;

    GPU_RQTImplementation( std::vector<Point>& points, int capacity):
        points(points),
        capacity(capacity)
    {}

    virtual ~GPU_RQTImplementation(){}

    virtual void allocate( int maxSize )
    {
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
    }

    virtual void insert_derived( int i, const Point& x )
    {
        quadtree::HyperRect<float,7> cell = bounds;
        quadtree::regular_insert<float>(
            points,i,cell,&root,capacity );
    }

    virtual void findNearest_derived( const Point& q )
    {
        touched = 0;
        queue.clear();
        result.clear();

        std::copy( q.data() + 0,
                   q.data() + 3, gpu_x_query.point );
        std::copy( q.data() + 3,
                   q.data() + 4, gpu_q_query.point );

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

                for( auto& pair : node->children )
                {
                    int x_idx = (pair.first & 0x07) ;
                    int q_idx = (pair.first & 0x78) >> 3;

                    pqueue_insert( queue,
                            quadtree::PriorityNode<float,7>(
                                gpu_x_result[x_idx]
                                    +w*gpu_q_result[q_idx],
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


Implementation* impl_gpu_rqt(std::vector<Point>& points, int capacity )
{
    return new GPU_RQTImplementation(points,capacity);
}


} //< namespace so3
} //< namespace robot_nn
} //< namespace mpblocks















#endif // SO3_H_
