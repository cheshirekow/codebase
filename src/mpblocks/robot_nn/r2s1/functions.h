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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/functions.h
 *
 *  @date   Nov 4, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_ROBOT_NN_FUNCTIONS_H_
#define MPBLOCKS_ROBOT_NN_FUNCTIONS_H_

#include <iostream>
#include <set>
#include <vector>

#include "BinaryKey.h"

namespace mpblocks {
namespace robot_nn {

template < typename Scalar, int NDim >
struct KDStructs
{



struct HyperRect
{
    Eigen::Matrix<Scalar,NDim,1> data[2];
    HyperRect()
    {
        data[0].fill(0);
        data[1].fill(0);
    }

    Eigen::Matrix<Scalar,NDim,1>& operator[]( int i )
    {
        return data[i];
    }

    const Eigen::Matrix<Scalar,NDim,1>& operator[]( int i ) const
    {
        return data[i];
    }
};


struct Node
{
    // interior node stuff
    int     idx;
    Scalar  value;
    Node*   child[2];

    // leaf node stuff
    std::vector<int> points;

    Node():
        idx(0),
        value(0)
    {
        child[0] = 0;
        child[1] = 0;
    }
};



typedef Eigen::Matrix<Scalar,NDim,1>  Point;


struct IsLeafFn
{
    bool operator()( Node* node ) const
    {
        return (node->child[0] == 0);
    }
};

struct IdxFn
{
    int operator()( Node* node ) const
    {
        return node->idx;
    }
};

struct ValueFn
{
    Scalar operator()( Node* node ) const
    {
        return node->value;
    }
};




struct LeafInsertFn
{
    std::vector<Node>&  nodes;
    std::vector<Point>& points;
    int                 capacity;

    LeafInsertFn(
            std::vector<Node>& nodes,
            std::vector<Point>& points,
            int capacity = 2 ):
        nodes(nodes),
        points(points),
        capacity(capacity)
    {}

    void operator()( const HyperRect& cell, Node* node, int point ) const
    {
        for(int i=0; i < 3; i++)
        {
            assert( cell[0][i] <= points[point][i]);
            assert( cell[1][i] >= points[point][i]);
        }

        node->points.push_back(point);
        if( node->points.size() > capacity )
        {
            // create two children nodes
            for(int i=0; i < 2; i++)
            {
                nodes.push_back(Node());
                node->child[i] = &nodes.back();
                node->child[i]->idx = (node->idx + 1) % NDim;
                node->child[i]->points.reserve(capacity+1);
            }

//            // get an array of the i'th value of all the points in our
//            // point set
//            std::vector<Scalar> vals;
//            vals.reserve( node->points.size() );
//            for( int pointIdx : node->points )
//                vals.push_back( points[pointIdx][node->idx] );
//
//            // find the median in the idx direction
//            int n = vals.size() / 2;
//            std::nth_element(vals.begin(), vals.begin()+n, vals.end() );
//            node->value = vals[n];
            node->value = (cell.data[0][node->idx]
                            + cell.data[1][node->idx])/2.0;

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
};

static Scalar r2s1Dist( Scalar weight, const Point& a, const Point& b )
{
    Scalar dist_2 = 0;
    Scalar dist_i = 0;
    for(int i=0; i < 2; i++)
    {
        dist_i = a[i] - b[i];
        dist_2 += dist_i*dist_i;
    }

    Scalar dist_theta;
    if( a[2] > b[2] )
        dist_theta = a[2] - b[2];
    else
        dist_theta = b[2]-a[2];
    dist_theta = std::min(dist_theta, 2*Scalar(M_PI)-dist_theta);

    return std::sqrt(dist_2) + weight*dist_theta;
}



struct ScheduledLeafInsertFn
{
    std::vector<Node>&  nodes;
    std::vector<Point>& points;
    Scalar              weight;
    int                 capacity;

    ScheduledLeafInsertFn(
            std::vector<Node>& nodes,
            std::vector<Point>& points,
            Scalar weight=Scalar(1.0),
            int capacity = 2 ):
        nodes(nodes),
        points(points),
        weight(weight),
        capacity(capacity)
    {}

    void operator()( const HyperRect& cell, Node* node, int point ) const
    {
        node->points.push_back(point);
        if( node->points.size() > capacity )
        {
            // find the longest cell face
            Scalar maxLength = 0;
            int    maxIdx    = 0;
            for(int i=0; i < 2; i++)
            {
                Scalar length   = cell.data[MAX][i] -
                                  cell.data[MIN][i];
                if( length > maxLength )
                {
                    maxIdx    = i;
                    maxLength = length;
                }
            }

            if(false)
            {
                Scalar length   = weight*(
                                   cell.data[MAX][2] -
                                   cell.data[MIN][2] );
                if( length > maxLength )
                    maxIdx = 2;
            }
            node->idx = maxIdx;

            // create two children nodes
            for(int i=0; i < 2; i++)
            {
                nodes.push_back(Node());
                node->child[i] = &nodes.back();
                node->child[i]->idx = (node->idx + 1) % NDim;
            }

            // get an array of the i'th value of all the points in our
            // point set
            std::vector<Scalar> vals;
            vals.reserve( node->points.size() );
            for( int pointIdx : node->points )
                vals.push_back( points[pointIdx][node->idx] );

            // find the median in the idx direction
            int n = vals.size() / 2;
            std::nth_element(vals.begin(), vals.begin()+n, vals.end() );
            node->value = vals[n];

            // reserve space for child point stores
            for(int i=0; i < 2; i++)
                node->child[i]->points.reserve(capacity+1);

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
};



struct UniformLeafInsertFn
{
    std::vector<Node>&  nodes;
    std::vector<Point>& points;
    Scalar              weight;
    int                 capacity;

    UniformLeafInsertFn(
            std::vector<Node>& nodes,
            std::vector<Point>& points,
            Scalar weight=Scalar(1.0),
            int capacity = 2 ):
        nodes(nodes),
        points(points),
        weight(weight),
        capacity(capacity)
    {}

    void operator()( const HyperRect& cell, Node* node, int point ) const
    {
        node->points.push_back(point);
        if( node->points.size() > capacity )
        {
            // find the longest cell face
            Scalar maxLength = 0;
            int    maxIdx    = 0;
            for(int i=0; i < 2; i++)
            {
                Scalar length   = cell.data[MAX][i] -
                                  cell.data[MIN][i];
                if( length > maxLength )
                {
                    maxIdx    = i;
                    maxLength = length;
                }
            }

            if( maxLength > weight*M_PI )
                node->idx = maxIdx;
            else
            {
                Scalar length   = weight*(
                                   cell.data[MAX][2] -
                                   cell.data[MIN][2] );
                if( length > maxLength )
                    node->idx = 2;
                else
                    node->idx = maxIdx;
            }

            // create two children nodes
            for(int i=0; i < 2; i++)
            {
                nodes.push_back(Node());
                node->child[i] = &nodes.back();
                node->child[i]->idx = (node->idx + 1) % NDim;
            }

            // get an array of the i'th value of all the points in our
            // point set
            std::vector<Scalar> vals;
            vals.reserve( node->points.size() );
            for( int pointIdx : node->points )
                vals.push_back( points[pointIdx][node->idx] );

            // find the median in the idx direction
            int n = vals.size() / 2;
            std::nth_element(vals.begin(), vals.begin()+n, vals.end() );
            node->value = vals[n];
//            node->value = 0.5*(cell[0][node->idx] + cell[1][node->idx]);

            // reserve space for child point stores
            for(int i=0; i < 2; i++)
                node->child[i]->points.reserve(capacity+1);

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
};




struct ChildFn
{
    Node* operator()( Node* node, int which ) const
    {
        return node->child[which];
    }
};

struct PointGetFn
{
    std::vector<Point>& points;

    PointGetFn( std::vector<Point>& points ):
        points(points)
    {}

    Scalar operator()( int point, int i ) const
    {
        return points[point][i];
    }
};

struct HyperSetFn
{
    void operator()( HyperRect& cell, int bound, int i, Scalar value ) const
    {
        cell.data[bound][i] = value;
    }
};


struct PriorityNode
{
    Scalar      dist;
    Node*       node;
    HyperRect   hrect;

    PriorityNode():
        dist(0),
        node(0)
    {}

    PriorityNode( Scalar dist, Node* node, const HyperRect& hrect ):
        dist(dist),
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

typedef std::vector<PriorityNode>                 Queue;
typedef std::set<ResultNode,
        typename ResultNode::LessThan>            ResultSet;


struct QueueMinKeyFn
{
    Scalar operator()( const Queue& q ) const
    {
        return q.front().dist;
    }
};

struct QueuePopMinFn
{
    PriorityNode operator()( Queue& q ) const
    {
        std::pop_heap(q.begin(),q.end(),
                        typename PriorityNode::GreaterThan());
        PriorityNode returnMe = q.back();
        q.pop_back();
        return returnMe;
    }
};

struct QueueInsertFn
{
    int& insertCount;

    QueueInsertFn( int& insertCount ):
        insertCount(insertCount)
    {}

    void operator()( Queue& q, Scalar dist, const PriorityNode& pair ) const
    {
        insertCount++;
        assert(pair.node);
        q.push_back( PriorityNode(dist,pair.node,pair.hrect) );
        std::push_heap(q.begin(),q.end(),
                        typename PriorityNode::GreaterThan());
    }
};

struct QueueSizeFn
{
    size_t operator()( const Queue& q ) const
    {
        return q.size();
    }
};

struct SetMaxKeyFn
{
    Scalar operator()( const ResultSet& r ) const
    {
        typename ResultSet::iterator back = r.end();
        back--;
        return back->dist;
    }
};

struct SetPopMaxFn
{
    void operator()( ResultSet& r ) const
    {
        typename ResultSet::iterator back = r.end();
        back--;
        r.erase(back);
    }
};

struct SetInsertFn
{
    void operator()( ResultSet& r, Scalar dist, int pointIdx ) const
    {
        r.insert( ResultNode(dist,pointIdx) );
    }
};

struct SetSizeFn
{
    size_t operator()( const ResultSet& r ) const
    {
        return r.size();
    }
};

struct PointDistFn
{
    std::vector<Point>& points;

    PointDistFn( std::vector<Point>& points ):
        points(points)
    {}

    Scalar operator()( const Point& q, int b ) const
    {
        return ( q - points[b] ).norm();
    }
};

struct R2S1PointDistFn
{
    std::vector<Point>& points;
    Scalar weight;

    R2S1PointDistFn( std::vector<Point>& points, Scalar weight = Scalar(1.0) ):
        points(points),
        weight(weight)
    {}

    Scalar operator()( const Point& q, int b ) const
    {
        return r2s1Dist(weight,q,points[b]);
    }
};

struct CellDistFn
{
    Scalar operator()( const Point& q, const HyperRect& h ) const
    {
        Scalar dist_2  = 0;
        Scalar dist_i = 0;
        for( int i=0; i < NDim; i++)
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
};

struct R2S1CellDistFn
{
    Scalar weight;

    R2S1CellDistFn( Scalar weight = Scalar(1.0) ):
        weight(weight)
    {}

    Scalar operator()( const Point& q, const HyperRect& h ) const
    {
        Scalar dist_2  = 0;
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
};



struct ChildrenFn
{
    std::vector<PriorityNode> operator()( const PriorityNode& node ) const
    {
        std::vector<PriorityNode> result;
        result.reserve(2);
        result.push_back( PriorityNode(0,node.node->child[0],node.hrect) );
        result.push_back( PriorityNode(0,node.node->child[1],node.hrect) );

        result[0].hrect.data[MAX][node.node->idx] = node.node->value;
        result[1].hrect.data[MIN][node.node->idx] = node.node->value;

        return result;
    }
};

struct SitesFn
{
    std::vector<int>& operator()( Node* node ) const
    {
        return node->points;
    }
};

struct CellFn
{
    const HyperRect& operator()( const PriorityNode& node ) const
    {
        return node.hrect;
    }
};

struct NodeFn
{
    Node* operator()( const PriorityNode& node ) const
    {
        return node.node;
    }
};





};




} //< namespace robot_nn
} //< namespace mpblocks















#endif // FUNCTIONS_H_
