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
#include "impl/ctors.h"
#include <mpblocks/dubins/curves_eigen.hpp>

namespace mpblocks {
namespace robot_nn {
namespace   dubins {

namespace mpd = mpblocks::dubins::curves_eigen;

/// regular quad trees
struct SKDImplementation:
    public Implementation
{
    typedef Eigen::Matrix<float,3,1>     Point;
    typedef mpd::hyper::HyperRect<float> HyperRect;

    struct Node
    {
        // interior node stuff
        int     idx;
        int     depth;
        float   value;
        Node*   child[2];

        // leaf node stuff
        std::vector<int> points;

        Node():
            idx(-1),
            depth(-1),
            value(-2000)
        {
            child[0] = 0;
            child[1] = 0;
        }
    };


    struct PriorityKey
    {
        float       dist;
        Node*       node;
        HyperRect   hrect;

        PriorityKey():
            dist(0),
            node(0)
        {}

        PriorityKey( float dist, Node* node, const HyperRect& hrect ):
            dist(dist),
            node(node),
            hrect(hrect)
        {
            assert(node);
        }

        struct GreaterThan
        {
            bool operator()( const PriorityKey& a, const PriorityKey& b )
            {
                if( a.dist > b.dist )
                    return true;
                else if( a.dist < b.dist )
                    return false;
                else
                    return (a.node > b.node);
            }
        };
    };

    struct ResultKey
    {
        float dist;
        int   point;

        ResultKey():
            dist(0),
            point(0)
        {}

        ResultKey( float dist, int point ):
            dist(dist),
            point(point)
        {}

        struct LessThan
        {
            bool operator()( const ResultKey& a, const ResultKey& b )
            {
                if( a.dist < b.dist )
                    return true;
                else if( a.dist > b.dist )
                    return false;
                else
                    return a.point < b.point;
            }
        };
    };

    std::vector<Point>&      points;
    std::vector<ResultKey>   result; //< max heap
    std::vector<PriorityKey> queue;  //< min heap

    PriorityKey::GreaterThan queue_compare;
    ResultKey::LessThan      result_compare;
    Node                     root;
    HyperRect                bounds;
    int                      capacity;
    int                      rate;

    SKDImplementation( std::vector<Point>& points, int capacity, int rate):
        points(points),
        capacity(capacity),
        rate(rate)
    {
        result.reserve(k+1);
        root.idx = 0;
        root.depth = 0;
    }

    virtual ~SKDImplementation(){}

    virtual void allocate( int maxSize )
    {
        for(int j=0; j < 2; j++)
        {
            bounds.minExt[j] = 0;
            bounds.maxExt[j] = ws;
        }
        bounds.minExt[2] = -float(M_PI);
        bounds.maxExt[2] = float(M_PI);
    }

    virtual void insert_derived( int i, const Point& x )
    {
        HyperRect cell = bounds;
        Node* node     = &root;

        int depth = 0;
        while( node->child[0] !=0 )
        {
            int idx  = node->idx;
            if( x[idx] <= node->value )
            {
                cell.maxExt[idx] = node->value;
                node = node->child[0];
            }
            else
            {
                cell.minExt[idx] = node->value;
                node = node->child[1];
            }
            depth++;
        }

        node->points.push_back(i);
        if( node->points.size() > capacity )
        {
            if(node->depth > 5)
                node->idx = node->depth % 3;
            else
                node->idx = node->depth % 2;
            // create two children nodes
            for(int j=0; j < 2; j++)
            {
                node->child[j] = new Node();
                node->child[j]->depth = (node->depth + 1);
                node->child[j]->points.reserve(capacity+1);
            }

            node->value = 0.5*(cell.minExt[node->idx]
                             + cell.maxExt[node->idx]);

            // split the point list into the appropriate child
            for( int pointIdx : node->points )
            {
                if( points[pointIdx][node->idx] <= node->value )
                    node->child[0]->points.push_back(pointIdx);
                else
                    node->child[1]->points.push_back(pointIdx);
            }

            // clear out our child array of the now interior node
            std::vector<int> dummy;
            node->points.swap(dummy);
        }
    }

    virtual void findNearest_derived( const Point& q )
    {
        touched = 0;
        queue.clear();
        result.clear();
        touched = 0;
        HyperRect cell = bounds;
        queue.push_back( PriorityKey(0,&root,cell) );

        while( queue.size() > 0 &&
                (result.size() < k
                        || queue[0].dist < result[0].dist ) )
        {
            touched++;

            // at each iteration expand the unsearched cell which contains the
            // nearest point to the query
            PriorityKey pair = queue[0];
            std::pop_heap(queue.begin(),queue.end(),queue_compare);
            queue.pop_back();

            Node*        node = pair.node;

            // if it's a leaf node then evaluate the distance to each site
            // contained in that nodes cell
            if( node->child[0] == 0 )
            {
                for( int site : node->points )
                {
                    const Point& x = points[site];
                    mpd::Path<float> dubins_result = mpd::solve(q, x, m_r );
                    result.push_back(
                            ResultKey(dubins_result.dist(m_r),site) );
                    std::push_heap(result.begin(),result.end(),result_compare);
                    if( result.size() > k )
                    {
                        std::pop_heap(
                                result.begin(),result.end(),result_compare);
                        result.pop_back();
                    }
                }
            }
            else
            {
                for( int i=0 ; i < 2; i++ )
                {
                    HyperRect cell = pair.hrect;
                    if( i==0 )
                        cell.maxExt[pair.node->idx] = pair.node->value;
                    else
                        cell.minExt[pair.node->idx] = pair.node->value;
                    mpd::Path<float> dubins_result =
                            mpd::hyper::solve(q,cell,m_r);
                    queue.push_back( PriorityKey(
                            dubins_result.dist(m_r),
                            pair.node->child[i],cell) );
                    std::push_heap(queue.begin(),queue.end(),queue_compare);
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


Implementation* impl_cpu_skd(std::vector<Point>& points, int capacity, int rate )
{
    return new SKDImplementation(points,capacity,rate);
}

}
}
}



#endif // KD2_H_
