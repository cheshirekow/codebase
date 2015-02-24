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
#include "BinaryKey.h"
#include "impl/ctors.h"

namespace mpblocks {
namespace robot_nn {



struct UKDImplementation:
    public Implementation
{

    struct HyperRect
    {
        Eigen::Matrix<float,3,1> data[2];
        HyperRect()
        {
            data[0].fill(0);
            data[1].fill(0);
        }

        Eigen::Matrix<float,3,1>& operator[]( int i )
        {
            return data[i];
        }

        const Eigen::Matrix<float,3,1>& operator[]( int i ) const
        {
            return data[i];
        }
    };


    struct Node
    {
        // interior node stuff
        int     idx;
        float   value;
        Node*   child[2];

        // leaf node stuff
        std::vector<int> points;

        Node():
            idx(-1),
            value(-2000)
        {
            child[0] = 0;
            child[1] = 0;
        }
    };


    struct PriorityNode
    {
        float       dist;
        Node*       node;
        HyperRect   hrect;

        PriorityNode():
            dist(0),
            node(0)
        {}

        PriorityNode( float dist, Node* node, const HyperRect& hrect ):
            dist(dist),
            node(node),
            hrect(hrect)
        {
            assert(node);
        }

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
        float   dist;
        int     point;

        ResultNode():
            dist(0),
            point(0)
        {}

        ResultNode( float dist, int point ):
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


    static float r2s1Dist( float weight, const Point& a, const Point& b )
    {
        float dist_2 = 0;
        float dist_i = 0;
        for(int i=0; i < 2; i++)
        {
            dist_i = a[i] - b[i];
            dist_2 += dist_i*dist_i;
        }

        float dist_theta;
        if( a[2] > b[2] )
            dist_theta = a[2] - b[2];
        else
            dist_theta = b[2]-a[2];
        dist_theta = std::min(dist_theta, 2*float(M_PI)-dist_theta);

        return std::sqrt(dist_2) + weight*dist_theta;
    }

    static float r2s1Dist( float weight, const Point& q, const HyperRect& h )
    {
        float dist_2  = 0;
        float dist_i = 0;

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

        float dist_theta = 2*float(M_PI);
        if( q[2] < h.data[MIN][2] )
        {
            float dist_a = h.data[MIN][2] - q[2];
            float dist_b = 2*float(M_PI) - (h.data[MAX][2] - q[2]);
            dist_theta = std::min(dist_a,dist_b);
        }
        else
        {
            float dist_a = q[2] - h.data[MAX][2];
            float dist_b = 2*float(M_PI) - (q[2] - h.data[MIN][2]);
            dist_theta = std::min(dist_a,dist_b);
        }

        return std::sqrt(dist_2) + weight*dist_theta;

        return std::sqrt(dist_2);
    }

    std::vector<Point>&         points;
    int                         capacity;
    Node                        root;
    std::vector<PriorityNode>   queue;
    std::vector<ResultNode>     result;
    HyperRect                   bounds;

    UKDImplementation( std::vector<Point>& points, int capacity ):
        points(points),
        capacity(capacity)
    {
        root.idx = 0;
    }

    virtual ~UKDImplementation(){}

    virtual void allocate( int maxSize )
    {
        for(int j=0; j < 2; j++)
        {
            bounds.data[robot_nn::MAX][j] = ws;
            bounds.data[robot_nn::MIN][j] = 0;
        }
        bounds.data[robot_nn::MIN][2] = -float(M_PI);
        bounds.data[robot_nn::MAX][2] = float(M_PI);
    }

    virtual void insert_derived( int i, const Point& x )
    {
        for(int j=0; j < 2; j++)
        {
            assert( bounds.data[robot_nn::MAX][j] == ws );
            assert( bounds.data[robot_nn::MIN][j] == 0 );
        }
        assert( bounds.data[robot_nn::MIN][2] == -float(M_PI) );
        assert( bounds.data[robot_nn::MAX][2] == float(M_PI) );

        HyperRect cell = bounds;
        Node* node     = &root;

        int depth = 0;
        while( node->child[0] !=0 )
        {
            int idx = node->idx;
            assert(idx < 3);
            int which = (x[idx] <= node->value) ? 0 : 1;

            cell.data[1-which][idx] = node->value;
            for(int j=0; j < 2; j++)
            {
                assert( cell.data[1][j] > cell.data[0][j] );
                assert( x[j] >= cell.data[0][j]);
                assert( x[j] <= cell.data[1][j]);
            }

            node = node->child[which];
            depth++;
        }

        // find the longest cell face
        float maxLength = 0;
        int    maxIdx    = 0;
        for(int i=0; i < 2; i++)
        {
            float length   = cell.data[MAX][i] -
                              cell.data[MIN][i];
            if( length > maxLength )
            {
                maxIdx    = i;
                maxLength = length;
            }
        }

        {
            float length = w*( cell.data[MAX][2] -
                               cell.data[MIN][2] );
            if( length > maxLength )
                maxIdx = 2;
        }
        node->idx = maxIdx;

        assert( !node->child[1] );
        node->points.push_back(i);
        if( node->points.size() > capacity )
        {
            // create two children nodes
            for(int j=0; j < 2; j++)
            {
                node->child[j] = new Node();
                node->child[j]->idx = (node->idx + 1) % 3;
                node->child[j]->points.reserve(capacity+1);
            }

            node->value = 0.5*(cell.data[0][node->idx]
                            + cell.data[1][node->idx]);
            assert(node->idx == 2 || node->value != float(0) );

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
        queue.push_back( PriorityNode(0,&root,cell) );

        while( queue.size() > 0 &&
                (result.size() < k
                        || queue[0].dist < result[0].dist ) )
        {
            touched++;

            // at each iteration expand the unsearched cell which contains the
            // nearest point to the query
            PriorityNode    pair = queue[0];
            std::pop_heap(queue.begin(),queue.end(),
                            PriorityNode::GreaterThan());
            queue.pop_back();

            Node*        node = pair.node;

            // if it's a leaf node then evaluate the distance to each site
            // contained in that nodes cell
            if( node->child[0] == 0 )
            {
                for( int site : node->points )
                {
                    result.push_back(
                            ResultNode(r2s1Dist(w,q,points[site]),site) );
                    std::push_heap(result.begin(),result.end(),
                                    ResultNode::LessThan());
                    if( result.size() > k )
                    {
                        std::pop_heap(result.begin(),result.end(),
                                        ResultNode::LessThan() );
                        result.pop_back();
                    }
                }
            }
            else
            {
                for( int i=0 ; i < 2; i++ )
                {
                    HyperRect cell = pair.hrect;
                    cell[1-i][pair.node->idx] = pair.node->value;
                    queue.push_back( PriorityNode(
                            r2s1Dist(w,q,cell),
                            pair.node->child[i],cell) );
                    std::push_heap(queue.begin(),queue.end(),
                            PriorityNode::GreaterThan());
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

Implementation* impl_ukd(std::vector<Point>& points, int capacity )
{
    return new UKDImplementation(points,capacity);
}


}
}



#endif // KD2_H_
