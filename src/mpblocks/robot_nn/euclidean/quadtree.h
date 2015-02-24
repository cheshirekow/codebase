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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/quadtree.h
 *
 *  @date   Nov 5, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_ROBOT_NN_QUADTREE_H_
#define MPBLOCKS_ROBOT_NN_QUADTREE_H_

#include <Eigen/Dense>
#include <bitset>
#include <mpblocks/redblack.h>
#include "BinaryKey.h"

namespace mpblocks {
namespace robot_nn {
namespace quadtree {



template <typename Scalar, int NDim>
using Point = Eigen::Matrix<Scalar,NDim,1>;



template< typename Scalar, int NDim>
struct HyperRect
{
    Eigen::Matrix<Scalar,NDim,1> data[2];
    HyperRect()
    {
        data[0].fill(0);
        data[1].fill(0);
    }
};

struct Node;
typedef std::map<int,Node*> ChildMap;

enum NodeType
{
    LEAF,
    INTERIOR
};

struct Node
{
    NodeType type;

    // pointers for interior nodes
    ChildMap children;

    // point array for leaf nodes
    std::vector<int> points;

    Node():
        type(LEAF)
    {}
};



template< typename Scalar, int IdxOff, int IdxDim,  int PointDim >
void get_idx( const Point<Scalar,PointDim>& point,
              const Point<Scalar,PointDim>& median,
              std::bitset<IdxDim>& bits )
{
    // get the index of the appropriate child
    for(int i=0; i < IdxDim; i++)
        bits[i] = point[IdxOff + i] > median[IdxOff + i];
}


template< typename Scalar, int IdxOff, int IdxDim,  int PointDim >
void refine( HyperRect<Scalar,PointDim>& cell,
              const Point<Scalar,PointDim>& median,
              std::bitset<IdxDim>& bits )
{
    for(int i=0; i < IdxDim; i++)
    {
        if( bits[i] )
            cell.data[MIN][IdxOff + i] = median[IdxOff + i];
        else
            cell.data[MAX][IdxOff + i] = median[IdxOff + i];
    }
}


template< typename Scalar, int IdxOff, int IdxDim, int PointDim >
void step_down( HyperRect<Scalar,PointDim>& cell,
                const Point<Scalar,PointDim>& point,
                Node*& node )
{
    // find the center of the current cell
    Point<Scalar,PointDim> median =
                    Scalar(0.5)*(cell.data[MAX] + cell.data[MIN] );

    // determine the orthant that the query falls in
    std::bitset<IdxDim> bits;
    get_idx<Scalar,IdxOff,IdxDim,PointDim>(point,median,bits);

    // refine the hyper rectangle for that orthant
    refine<Scalar,IdxOff,IdxDim,PointDim>(cell,median,bits);

//    std::cout << "step down to cell:"
//              << "\n   " << cell.data[MIN].transpose()
//              << "\n   " << cell.data[MAX].transpose()
//              << "\n";

    // get the index of the child for that orthant
    int idx = bits.to_ulong();

    // get the child for that index
    typename ChildMap::iterator iter = node->children.find(idx);

    // if the child slot is empty then create a new node for the child
    if( iter == node->children.end() )
    {
        Node* newNode = new Node();
        node->children[idx] = newNode;
        node = newNode;
    }
    else
        node = iter->second;
}

template< typename Scalar, int IdxOff, int IdxDim, int PointDim, class PointDeref >
void split( const PointDeref& deref,
            HyperRect<Scalar,PointDim>& cell,
            Node* node,
            int capacity
            )
{
    Point<Scalar,PointDim> median =
                Scalar(0.5)* (cell.data[MAX] + cell.data[MIN]);
    std::bitset<IdxDim> bits;
    for( int pointRef : node->points )
    {
        // get the index of the child for the orthant of this point
        get_idx<Scalar,IdxOff,IdxDim,PointDim>( deref[pointRef], median, bits );
        int idx = bits.to_ulong();

        // get the child for that index
        Node* child = 0;
        typename ChildMap::iterator iter = node->children.find(idx);

        // if the child slot is empty then create a new node for the child
        if( iter == node->children.end() )
        {
            child = new Node();
            child->points.reserve(capacity);
            node->children[idx] = child;
        }
        else
            child = iter->second;

        // insert this point into that child
        child->points.push_back(pointRef);
    }

    // clear out the parent node's points
    std::vector<int> dummy;
    node->points.swap(dummy);

    // mark the parent as interior
    node->type = INTERIOR;
}


enum
{
    INDEX_R2,
    INDEX_R1
};

struct ScheduleItem
{
    int depth;
    int index;

    ScheduleItem():
        depth(0),
        index(0)
    {}

    ScheduleItem( int depth, int index ):
        depth(depth),
        index(index)
    {}

    struct LessThan
    {
        bool operator()( const ScheduleItem& a, const ScheduleItem& b )
        {
            return a.depth < b.depth;
        }
    };
};


void build_schedule( std::vector<ScheduleItem>& schedule,
                     float s0, float w, int rate, int depth );

template< typename Scalar, int NDim >
void regular_insert(
        const std::vector< Point<Scalar,NDim> >& points,
        int pointRef,
        HyperRect<Scalar,NDim>& cell,
        Node* node,
        int capacity )
{
//    std::cout << "Inserting: " << points[pointRef].transpose() << "\n";
    while( node->type == INTERIOR )
        step_down<Scalar,0,NDim,NDim>(cell,points[pointRef],node);

    // if the node is newly created then reserve space for it's point set
    if( node->points.size() == 0 )
        node->points.reserve( capacity );

    // insert the point into that node
    node->points.push_back(pointRef);

    // if the node is past capacity then split it
    if( node->points.size() > capacity )
    {
        split<Scalar,0,NDim,NDim>(points,cell,node,capacity);
    }
}


template< typename Scalar >
void scheduled_insert(
        const std::vector< Point<Scalar,3> >& points,
        int pointRef,
        HyperRect<Scalar,3>& cell,
        Node* node,
        int capacity,
        ScheduleItem* schedule )
{
//    std::cout << "Inserting: " << points[pointRef].transpose() << "\n";

    int depth = 0;
    while( node->type == INTERIOR )
    {
        depth++;
        while( schedule->depth < depth )
            schedule++;

        if( schedule->index == INDEX_R2 )
            step_down<Scalar,0,2,3>(cell,points[pointRef],node);
        else
            step_down<Scalar,2,1,3>(cell,points[pointRef],node);
    }

    // if the node is newly created then reserve space for it's point set
    if( node->points.size() == 0 )
        node->points.reserve( capacity );

    // insert the point into that node
    node->points.push_back(pointRef);

    // if the node is past capacity then split it
    if( node->points.size() > capacity )
    {
        depth++;
        while( schedule->depth < depth )
            schedule++;

        if( schedule->index == INDEX_R2 )
            split<Scalar,0,2,3>(points,cell,node,capacity);
        else
            split<Scalar,2,1,3>(points,cell,node,capacity);
    }
}


template <typename Scalar, int PointDim>
struct PriorityNode
{
    Scalar              dist;
    int                 depth;
    Node*               node;
    HyperRect<Scalar,PointDim> hrect;

    PriorityNode():
        dist(0),
        depth(0),
        node(0)
    {}

    PriorityNode( Scalar dist, int depth,
                    Node* node, const HyperRect<Scalar,PointDim>& hrect ):
        dist(dist),
        depth(depth),
        node(node),
        hrect(hrect)
    {
        assert(node);
    }

    struct LessThan
    {
        bool operator()( const PriorityNode& a, const PriorityNode& b )
        {
            if( a.dist < b.dist )
                return true;
            else if( a.dist > b.dist )
                return false;
            else
                return (a.node < b.node);
        }
    };

    struct GreaterThan
    {
        bool operator()( const PriorityNode& a, const PriorityNode& b )
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

template <typename Scalar>
struct ResultNode
{
    Scalar  dist;
    int     point;

    ResultNode():
        dist(0),
        point(0)
    {}

    ResultNode( Scalar dist, int point ):
        dist(dist),
        point(point)
    {}

    struct LessThan
    {
        bool operator()( const ResultNode& a, const ResultNode& b )
        {
            if( a.dist < b.dist )
                return true;
            else if( a.dist > b.dist )
                return false;
            else
                return (a.point < b.point);
        }
    };
};


template <typename Scalar, int PointDim>
Scalar distance( const Point<Scalar,PointDim>& q,
                 const HyperRect<Scalar,PointDim>& h )
{
    Scalar dist_2  = 0;
    Scalar dist_i = 0;
    for( int i=0; i < PointDim; i++)
    {
        if( q[i] < h.data[MIN][i] )
            dist_i = h.data[MIN][i] - q[i];
        else if( q[i] > h.data[MAX][i] )
            dist_i = q[i] - h.data[MAX][i];
        else
            dist_i = 0;
        dist_2 += dist_i*dist_i;
    }

    return std::sqrt(dist_2);
}

template <typename Scalar, int PointDim>
Scalar r2s1_distance(
        const Point<Scalar,PointDim>& q,
        const HyperRect<Scalar,PointDim>& h,
        Scalar weight )
{
    Scalar dist_2 = 0;
    Scalar dist_i = 0;

    for( int i=0; i < 2; i++)
    {
        if( q[i] < h.data[MIN][i] )
            dist_i = h.data[MIN][i] - q[i];
        else if( q[i] > h.data[MAX][i] )
            dist_i = q[i] - h.data[MAX][i];
        else
            dist_i = 0;
        dist_2 += dist_i*dist_i;
    }

    if( h.data[MIN][2] < q[2] && q[2] < h.data[MAX][2] )
        return std::sqrt(dist_2);

    Scalar dist_theta = 2*Scalar(M_PI);
    if( q[2] < h.data[MIN][2] )
    {
        Scalar dist_a = h.data[MIN][2] - q[2];
        Scalar dist_b = 2*Scalar(M_PI) - (h.data[MAX][2] - q[2]);
        dist_theta = std::min(dist_a,dist_b);
    }
    else
    {
        Scalar dist_a = q[2] - h.data[MAX][2];
        Scalar dist_b = 2*Scalar(M_PI) - (q[2] - h.data[MIN][2]);
        dist_theta = std::min(dist_a,dist_b);
    }

    return std::sqrt(dist_2) + weight*dist_theta;
}



template <typename Scalar, int PointDim>
Scalar distance( const Point<Scalar,PointDim>& q,
                 const Point<Scalar,PointDim>& b )
{
    return (q-b).norm();
}

template <typename Scalar, int PointDim>
Scalar r2s1_distance(
        const Point<Scalar,PointDim>& q,
        const Point<Scalar,PointDim>& b,
        Scalar weight )
{
    Scalar dist_2 = 0;
    Scalar dist_i = 0;

    for( int i=0; i < 2; i++)
    {
        dist_i = q[i] - b[i];
        dist_2 += dist_i*dist_i;
    }

    Scalar dist_theta = 2*Scalar(M_PI);
    if( q[2] < b[2] )
    {
        Scalar dist_a = b[2] - q[2];
        Scalar dist_b = 2*Scalar(M_PI) - (b[2] - q[2]);
        dist_theta = std::min(dist_a,dist_b);
    }
    else
    {
        Scalar dist_a = q[2] - b[2];
        Scalar dist_b = 2*Scalar(M_PI) - (q[2] - b[2]);
        dist_theta = std::min(dist_a,dist_b);
    }

    return std::sqrt(dist_2) + weight*dist_theta;
}


template <typename Scalar, int PointDim >
void pqueue_popmin(
        std::vector< PriorityNode<Scalar,PointDim> >& q,
       PriorityNode<Scalar,PointDim>& out )
{
    out = q[0];
    std::pop_heap(q.begin(),q.end(),
                typename PriorityNode<Scalar,PointDim>::GreaterThan() );
    q.pop_back();
}

template <typename Scalar, int PointDim >
void pqueue_insert(
        std::vector< PriorityNode<Scalar,PointDim> >& q,
       const PriorityNode<Scalar,PointDim>& in )
{
    q.push_back( in );
    std::push_heap(q.begin(),q.end(),
                typename PriorityNode<Scalar,PointDim>::GreaterThan() );
}

template <typename Scalar >
void result_popmax( std::vector< ResultNode<Scalar> >& q)
{
    std::pop_heap(q.begin(),q.end(),
                typename ResultNode<Scalar>::LessThan() );
    q.pop_back();
}

template <typename Scalar >
void result_insert( std::vector< ResultNode<Scalar> >& q,
                    const ResultNode<Scalar>& in )
{
    q.push_back( in );
    std::push_heap(q.begin(),q.end(),
                typename ResultNode<Scalar>::LessThan() );
}




} //< quadtree
} //< robot_nn
} //< mpblocks














#endif // QUADTREE_H_
