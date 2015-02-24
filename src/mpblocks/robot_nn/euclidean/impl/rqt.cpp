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


#include "config.h"
#include "Implementation.h"
#include "quadtree.h"
#include "impl/ctors.h"

namespace mpblocks {
namespace robot_nn {
namespace euclidean{


/// regular quad trees
template <int NDim>
struct RQTImplementation:
    public Implementation<NDim>
{
    typedef Eigen::Matrix<float,NDim,1> Point;
    std::vector<Point>& points;
    int                 capacity;

    quadtree::HyperRect<float,NDim> bounds;
    quadtree::Node                  root;
    std::vector< quadtree::PriorityNode<float,NDim> > queue;
    std::vector< quadtree::ResultNode<float> >        result;

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

        for(int j=0; j < NDim; j++)
        {
            bounds.data[robot_nn::MAX][j] = ws;
            bounds.data[robot_nn::MIN][j] = 0;
        }
//        std::cout << "Quadtree schedule:\n";
//        for( auto item : schedule )
//            printf("   %4d,%d\n",item.depth,item.index);
    }

    virtual void insert_derived( int i, const Point& x )
    {
        quadtree::HyperRect<float,NDim> cell = bounds;
        quadtree::regular_insert<float,NDim>(
            points,i,cell,&root,capacity );
    }

    virtual void findNearest_derived( const Point& q )
    {
        queue.clear();
        result.clear();

        quadtree::HyperRect<float,NDim> cell = bounds;
        quadtree::pqueue_insert(queue,
                quadtree::PriorityNode<float,NDim>(
                    quadtree::distance<float,NDim>(q,cell),
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
            quadtree::PriorityNode<float,NDim> pqnode;
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
                            quadtree::distance(q,points[pointRef])
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
                quadtree::Point<float,NDim> median =
                        0.5f * ( pqnode.hrect.data[MIN] +
                                 pqnode.hrect.data[MAX] );

                for( auto& pair : node->children )
                {
                    quadtree::HyperRect<float,NDim> cell = pqnode.hrect;
                    std::bitset<NDim> bits( pair.first );
                    quadtree::refine<float,0,NDim,NDim>( cell,median,bits );
                    pqueue_insert( queue,
                            quadtree::PriorityNode<float,NDim>(
                                quadtree::distance<float,NDim>(q,cell),
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


Implementation<2>* impl_rqt2(std::vector<Point2>& points, int capacity )
{
    return new RQTImplementation<2>(points,capacity);
}

Implementation<3>* impl_rqt3(std::vector<Point3>& points, int capacity )
{
    return new RQTImplementation<3>(points,capacity);
}

}
}
}


