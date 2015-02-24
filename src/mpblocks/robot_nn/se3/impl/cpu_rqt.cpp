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
#include "config.h"
#include "se3.h"
#include "Implementation.h"
#include "quadtree.h"
#include "impl/ctors.h"


namespace mpblocks {
namespace robot_nn {
namespace      se3 {

/// regular quad trees
struct RQTImplementation:
    public Implementation
{
    std::vector<Point>& points;
    int                 capacity;

    quadtree::HyperRect<float,7> bounds;
    quadtree::Node               root;
    std::vector< quadtree::PriorityNode<float,7> > queue;
    std::vector< quadtree::ResultNode<float> >     result;

//    KDImpl::Queue     queue;
//    KDImpl::ResultSet result;

    RQTImplementation( std::vector<Point>& points, int capacity):
        points(points),
        capacity(capacity)
    {}

    virtual ~RQTImplementation(){}

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

                for( auto& pair : node->children )
                {
                    quadtree::HyperRect<float,7> cell = pqnode.hrect;
                    std::bitset<7> bits( pair.first );
                    quadtree::refine<float,0,7,7>( cell,median,bits );

                    float dist;
                    bool  feas;
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


Implementation* impl_cpu_rqt(std::vector<Point>& points, int capacity )
{
    return new RQTImplementation(points,capacity);
}


} //< namespace so3
} //< namespace robot_nn
} //< namespace mpblocks















#endif // SO3_H_
