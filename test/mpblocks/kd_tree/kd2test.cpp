/*
 *  \file   CSampler.cpp
 *
 *  \date   Mar 26, 2011
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */


#include <vector>
#include <bitset>
#include <queue>
#include <set>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>


#include "kdtree.h"

#include <Eigen/Dense>
#include <mpblocks/kd2/insert.h>
#include <mpblocks/kd2/findNearest.h>

//#define TFMT    float
#define TFMT    double

#define NDIM        3
#define NTEST       100
#define NQUERY      10
#define CAPACITY    2

using namespace mpblocks;
using namespace kd2;


struct HyperRect
{
    Eigen::Matrix<TFMT,NDIM,1> data[2];
    HyperRect()
    {
        data[0].fill(0);
        data[1].fill(0);
    }
};

struct Node
{
    //< interior node stuff
    int     idx;
    TFMT    value;
    Node*   child[2];

    //< leaf node stuff
    std::vector<int> points;

    Node():
        idx(0),
        value(0)
    {
        child[0] = 0;
        child[1] = 0;
    }
};

typedef Eigen::Matrix<TFMT,NDIM,1>  Point;


std::vector<Node>       g_nodes;
std::vector<Point>      g_points;


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
    TFMT operator()( Node* node ) const
    {
        return node->value;
    }
};




struct LeafInsertFn
{
    void operator()( Node* node, int point ) const
    {
        node->points.push_back(point);
        if( node->points.size() > CAPACITY )
        {
            // create two children nodes
            for(int i=0; i < 2; i++)
            {
                g_nodes.push_back(Node());
                node->child[i] = &g_nodes.back();
                node->child[i]->idx = (node->idx + 1) % NDIM;
            }

            // get an array of the i'th value of all the points in our
            // point set
            std::vector<TFMT> vals;
            vals.reserve( node->points.size() );
            for( int pointIdx : node->points )
                vals.push_back( g_points[pointIdx][node->idx] );

            // find the median in the idx direction
            int n = vals.size() / 2;
            std::nth_element(vals.begin(), vals.begin()+n, vals.end() );
            node->value = vals[n];

            // reserve space for child point stores
            for(int i=0; i < 2; i++)
                node->child[i]->points.reserve(CAPACITY+1);

            // split the point list into the appropriate child
            for( int pointIdx : node->points )
            {
                if( g_points[pointIdx][node->idx] <= node->value )
                    node->child[0]->points.push_back(pointIdx);
                else
                    node->child[1]->points.push_back(pointIdx);
            }

            // clear out our child array
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
    TFMT operator()( int point, int i ) const
    {
        return g_points[point][i];
    }
};

struct HyperSetFn
{
    void operator()( HyperRect& cell, int bound, int i, TFMT value ) const
    {
        cell.data[bound][i] = value;
    }
};


struct PriorityNode
{
    TFMT        dist;
    Node*       node;
    HyperRect   hrect;

    PriorityNode():
        dist(0),
        node(0)
    {}

    PriorityNode( TFMT dist, Node* node, const HyperRect& hrect ):
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
    TFMT    dist;
    int     point;

    ResultNode():
        dist(0),
        point(0)
    {}

    ResultNode( TFMT dist, int point ):
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
typedef std::set<ResultNode,ResultNode::LessThan> ResultSet;


struct QueueMinKeyFn
{
    TFMT operator()( const Queue& q ) const
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
    void operator()( Queue& q, TFMT dist, const PriorityNode& pair ) const
    {
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
    TFMT operator()( const ResultSet& r ) const
    {
        ResultSet::iterator back = r.end();
        back--;
        return back->dist;
    }
};

struct SetPopMaxFn
{
    void operator()( ResultSet& r ) const
    {
        ResultSet::iterator back = r.end();
        back--;
        r.erase(back);
    }
};

struct SetInsertFn
{
    void operator()( ResultSet& r, TFMT dist, int pointIdx ) const
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
    TFMT operator()( const Point& q, int b ) const
    {
        return ( q - g_points[b] ).norm();
    }
};

struct CellDistFn
{
    TFMT operator()( const Point& q, const HyperRect& h ) const
    {
        TFMT dist_2  = 0;
        TFMT dist_i = 0;
        for( int i=0; i < NDIM; i++)
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


struct Dummy
{
    int dummy;
    Dummy( int idx ): dummy(idx){}
};


int main()
{

    using std::cout;
    using std::endl;

    srand(time(NULL));
    //srand(0);

    kdtree* kd_tree = kd_create(NDIM);
    g_points.resize(NTEST);
    g_nodes.reserve(2*NTEST);
    g_nodes.push_back( Node() );
    Node*   kd2_root = &g_nodes[0];

    HyperRect bounds;

    void*   data = 0;
    int     idx  = 0;
    TFMT    point[NDIM];

    for(unsigned int i=0; i < NTEST; i++)
    {
        for(unsigned int j=0; j < NDIM; j++)
        {
            point[j]  = rand()/(TFMT)RAND_MAX;
            g_points[i][j] = point[j];

            if( point[j] > bounds.data[MAX][j] )
                bounds.data[MAX][j] = point[j];
            if( point[j] < bounds.data[MIN][j] )
                bounds.data[MIN][j] = point[j];
        }

        data = static_cast<void*>( new Dummy(idx++) );
#ifdef FMT_IS_FLOAT
        kd_insertf(kd_tree,point,data);
#else
        kd_insert(kd_tree,point,data);
#endif

        kd2::insert( kd2_root, i,
                IsLeafFn(),
                LeafInsertFn(),
                IdxFn(),
                ValueFn(),
                ChildFn(),
                PointGetFn() );
    }

    cout << "Nearest Node Test:\n-------------------\n";

    Point ePoint;
    for(unsigned int i=0; i < NTEST; i++)
    {
        for(unsigned int j=0; j < NDIM; j++)
        {
            point[j] = rand()/(TFMT)RAND_MAX;
            ePoint[j]= point[j];
        }

#ifdef FMT_IS_FLOAT
        kdres*  kdResult = kd_nearestf(kd_tree,point);
#else
        kdres*  kdResult = kd_nearest(kd_tree,point);
#endif
        void*   cResult = kd_res_item_data(kdResult);
        Dummy*  cDummy  = static_cast<Dummy*>(cResult);
        int     cIdx    = cDummy->dummy;
        kd_res_free(kdResult);

        Queue       q;
        ResultSet   r;
        q.reserve( g_nodes.size() );
        q.push_back( PriorityNode(0,&g_nodes[0],bounds) );
        findNearest( ePoint, 1, q, r,
                QueueMinKeyFn(),
                QueuePopMinFn(),
                QueueInsertFn(),
                QueueSizeFn(),
                SetMaxKeyFn(),
                SetPopMaxFn(),
                SetInsertFn(),
                SetSizeFn(),
                PointDistFn(),
                CellDistFn(),
                IsLeafFn(),
                ChildrenFn(),
                SitesFn(),
                CellFn(),
                NodeFn() );

        ResultSet::iterator iter = r.begin();
        if( cIdx == iter->point )
            printf("   %4d: passed\n",i);
        else
            printf("   %4d: failed\n",i);
    }


    cout << endl;
    cout << "K Nearest Search Test:\n---------------------\n";

    for(unsigned int i=0; i < NTEST; i++)
    {
        for(unsigned int j=0; j < NDIM; j++)
        {
            point[j] = rand()/(TFMT)RAND_MAX;
            ePoint[j]= point[j];
        }

        for(unsigned int k=2; k < NTEST; k+=2)
        {
            Queue       q;
            ResultSet   r;
            q.reserve( g_nodes.size() );
            q.push_back( PriorityNode(0,&g_nodes[0],bounds) );
            findNearest( ePoint, k, q, r,
                    QueueMinKeyFn(),
                    QueuePopMinFn(),
                    QueueInsertFn(),
                    QueueSizeFn(),
                    SetMaxKeyFn(),
                    SetPopMaxFn(),
                    SetInsertFn(),
                    SetSizeFn(),
                    PointDistFn(),
                    CellDistFn(),
                    IsLeafFn(),
                    ChildrenFn(),
                    SitesFn(),
                    CellFn(),
                    NodeFn() );

            TFMT eps   = 1e-6;
            TFMT range = SetMaxKeyFn()(r) + eps;

#ifdef FMT_IS_FLOAT
            kdres*  kdResult = kd_nearest_rangef(kd_tree,point,range);
#else
            kdres*  kdResult = kd_nearest_range(kd_tree,point,range);
#endif
            bool passed = true;
            if(kd_res_size(kdResult) != r.size())
                passed = false;

            unsigned int cResultSize = kd_res_size(kdResult);
            while(!kd_res_end(kdResult))
            {
                void*   cResult = kd_res_item_data(kdResult);
                Dummy*  cDummy  = static_cast<Dummy*>(cResult);

                bool found = false;
                ResultSet::iterator iKey;

                for(iKey = r.begin(); iKey != r.end(); ++iKey)
                {
                    if( iKey->point == cDummy->dummy )
                    {
                        found = true;
                        break;
                    }
                }

                if(!found)
                {
                    passed = false;
                    break;
                }

                kd_res_next(kdResult);
            }

            kd_res_free(kdResult);

            if( passed )
                printf("   %4d,%3d: passed (%3d)\n",i,k,cResultSize);
            else
                printf("   %4d,%3d: failed\n",i,k);
        }
    }

    cout << endl;
    kd_free(kd_tree);

    return 0;
}
