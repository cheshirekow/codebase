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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/r2s1.cpp
 *
 *  @date   Nov 1, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */


#include <vector>
#include <map>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <set>

#include <Eigen/Dense>
#include <mpblocks/utility/Timespec.h>
#include <mpblocks/cudaNN/rect_dist.h>
#include <mpblocks/cuda.h>

#include "config.h"
#include "so3.h"
#include "quadtree.h"

const std::vector<int>  numSamplesList =
{
    0,
    10,
    100,
    200,
//    400,
//    1000,
//    2000,
//    4000,
//    10000,
//    20000,
//    40000,
//    100000,
//    200000,
//    400000,
//    800000,
//    1000000,
//    2000000,
//    4000000,
//    8000000,
//    10000000,
};

using namespace mpblocks;
using namespace robot_nn;
using namespace so3;

typedef Eigen::Vector4f Point;

void box_muller(float& oa, float& ob)
{
    float u1 = rand() / float(RAND_MAX);
    float u2 = rand() / float(RAND_MAX);

    float a = std::sqrt( float(-2)*std::log(u1) );
    float b = float(2*M_PI)*u2;

    oa = a*std::cos(b);
    ob = b*std::sin(b);
}

void uniform_so3( Point& q )
{
    box_muller( q[0], q[1] );
    box_muller( q[2], q[3] );
    q.normalize();
}

struct Result
{
    float dist;
    int   idx;

    Result( float dist, int idx ):
        dist(dist),
        idx(idx)
    {}

    Result():
        dist(1e9),
        idx(0)
    {}

    struct LessThan
    {
        bool operator()( const Result& a, const Result& b )
        {
            if( a.dist < b.dist )
                return true;
            else if( a.dist > b.dist )
                return false;
            else
                return a.idx < b.idx;
        }
    };
};

int main( int argc, char** argv )
{
    cuda::setDevice(0);
    cudaNN::so3_distance<true,float,4> cuda_dist;

    // spec check
    printf("Spec Check:\n--------------------\n");
    std::vector<bool> hasSpec(81,false);
    for( int i=0; i < 3; i++)
        for(int j=0; j < 3; j++)
            for(int k=0; k < 3; k++)
                for(int m=0; m < 3; m++)
                    hasSpec[ ConstraintSpec(i,j,k,m).toIdx() ] = true;

    for(int i=0; i < 81; i++)
        printf( "%3d : %s\n",i, (hasSpec[i] ? "PASS" : "FAIL") );
    printf("\n");

    std::vector< ConstraintSpec > specs;
    specs.reserve(81);
    for(int i=0; i < 81; i++)
        specs.push_back( ConstraintSpec(i) );
    hasSpec.clear();
    hasSpec.resize(81,false);
    for( auto& spec : specs )
        hasSpec[ spec.toIdx() ] = true;

    for(int i=0; i < 81; i++)
        printf( "%3d : %s\n",i, (hasSpec[i] ? "PASS" : "FAIL") );
    printf("\n");

    std::vector<std::string> constraintStr;
    constraintStr.reserve(128);
    constraintStr =
    {
        "OFF",
        "MIN",
        "MAX"
    };
    for(int i=3; i < 128; i++)
        constraintStr.push_back("INVALID");

    for(int i=0; i < 81; i++)
    {
        printf( "%3d : ",i );
        for(int j=0; j < 4; j++)
            printf(" %s ", constraintStr[ specs[i].m_data[j] ].c_str() );
        printf("\n");
    }
    printf("\n");

    utility::Timespec app_start,sw_start,sw_stop;
    utility::Timespec dist_start,dist_stop;
    utility::Timespec dist_cpu(0,0);
    utility::Timespec dist_gpu(0,0);

    std::vector<Point>    points;
    std::vector<Point>    queries;

    std::vector<int> result_truth;
    std::vector<int> result_hope;

    quadtree::Node                  quad_root;
    quadtree::HyperRect<float,4>    quad_bounds;

    for(int i=0; i < 4; i++)
    {
        quad_bounds[0][i] = -1.01;
        quad_bounds[1][i] = 1.01;
    }

    int maxSamples = 0;
    for(int numSamples : numSamplesList )
        if( numSamples > maxSamples )
            maxSamples = numSamples;

    points .resize( maxSamples,Point::Zero() );
    queries.resize( numQueries,Point::Zero() );

    std::cout << "Generating point set\n";
    //srand( time(NULL) );
    srand( 0 );

    double lastPrint = 0;
    clock_gettime(CLOCK_MONOTONIC,&app_start);
    for( int i=0; i < points.size(); i++ )
    {
        Point& point = points[i];
        uniform_so3(point);

        clock_gettime(CLOCK_MONOTONIC,&sw_stop);
        if( (sw_stop-app_start).milliseconds() - lastPrint > printTimeout  )
        {
            lastPrint += printTimeout;
            printf("   %6.2f%%\r",100.0*i/points.size());
            std::cout.flush();
        }
    }
    printf("   %6.2f%%\n",100.0);

    for( Point& point : queries )
        uniform_so3(point);

    for( int iNumSamples = 1;
            iNumSamples < numSamplesList.size(); iNumSamples++ )
    {
        int numSamples = numSamplesList[iNumSamples];
        int oldKdSize  = numSamplesList[iNumSamples-1];
        std::cout << "numSamples: " << numSamples << "\n";

        for(int i=oldKdSize; i<numSamples; i++)
        {
            quadtree::HyperRect<float,4> bounds = quad_bounds;
            quadtree::regular_insert<float,4>(
                        points, i,
                        bounds, &quad_root, 10 );


            clock_gettime(CLOCK_MONOTONIC,&sw_stop);
            if( (sw_stop-app_start).milliseconds() - lastPrint > printTimeout )
            {
                lastPrint += printTimeout;
                printf("   building tree: %6.2f%%\r",
                                100.0*(i-oldKdSize)/(numSamples-oldKdSize));
                std::cout.flush();
            }
        }
        printf("   building tree: %6.2f%%\n",100.0);


        for( int i=0; i < queries.size(); i++ )
        {
            const Point& q = queries[i];

            // brute force
            std::set<Result,Result::LessThan> bf_result;
            for( int j=0; j < numSamples; j++ )
            {
                const Point& x = points[j];
                bf_result.insert( Result (
                    so3_pseudo_distance(q,x),
                    j
                ) );

                if( bf_result.size() > k )
                {
                    auto iter = bf_result.end();
                    iter--;
                    bf_result.erase( iter );
                }
            }

            result_truth.clear();
            for( auto& item : bf_result )
                result_truth.push_back( item.idx );

            // quadtree
            std::vector< quadtree::PriorityNode<float,4> > queue;
            std::vector< quadtree::ResultNode<float> >     result;
            quadtree::HyperRect<float,4> bounds = quad_bounds;
            queue.push_back( quadtree::PriorityNode<float,4>(
                0,0,&quad_root,bounds) );

            while( queue.size() > 0 && ( result.size() < k ||
                    queue[0].dist < result[0].dist) )
            {
                quadtree::PriorityNode<float,4> pqnode;
                quadtree::pqueue_popmin(queue,pqnode);

                quadtree::Node* node = pqnode.node;
                // if the node is a leaf node, process all of it's entries
                if( node->type == quadtree::LEAF )
                {
                    for(int pointRef : node->points )
                    {
                        quadtree::result_insert( result,
                            quadtree::ResultNode<float>(
                                so3_pseudo_distance(q,points[pointRef])
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
                    Point median =
                            0.5f * ( pqnode.hrect[0] +
                                     pqnode.hrect[1] );

                    // use gpu to compute child distance
                    float child_dist[16];
                    std::map<int,float> cpu_child_dist;

                    cudaNN::RectangleQuery<float,4> cuda_rect;
                    std::copy( pqnode.hrect[0].data(),
                               pqnode.hrect[0].data() + 4, cuda_rect.min );
                    std::copy( pqnode.hrect[1].data(),
                               pqnode.hrect[1].data() + 4, cuda_rect.max );
                    std::copy( q.data(), q.data() + 4, cuda_rect.point );

                    clock_gettime( CLOCK_MONOTONIC, &dist_start );
                    cuda_dist(cuda_rect,child_dist);
                    clock_gettime( CLOCK_MONOTONIC, &dist_stop );
                    dist_gpu += (dist_stop - dist_start);
                    cuda::checkLastError("GPU rectangle distance\n");

                    for( auto& pair : node->children )
                    {
                        quadtree::HyperRect<float,4> cell = pqnode.hrect;
                        std::bitset<4> bits( pair.first );
                        quadtree::refine<float,0,4,4>( cell,median,bits );

                        clock_gettime( CLOCK_MONOTONIC, &dist_start );
                        float dist     = 1e9;
                        bool  feasible = false;
                        for(int i=0; i < 81; i++)
                        {
                            float dist_i     = 1e9;
                            bool  feasible_i = false;
                            so3_pseudo_distance<-1>(
                                    ConstraintSpec(i),q,cell,dist_i,feasible_i );
                            if( feasible_i && dist_i < dist )
                            {
                                dist = dist_i;
                                feasible = true;
                            }

                            so3_pseudo_distance<1>(
                                    ConstraintSpec(i),q,cell,dist_i,feasible_i );
                            if( feasible_i && dist_i < dist )
                            {
                                dist = dist_i;
                                feasible = true;
                            }
                        }
                        clock_gettime( CLOCK_MONOTONIC, &dist_stop );
                        dist_cpu += (dist_stop - dist_start);
                        cpu_child_dist[pair.first] = dist;

                        if(!feasible)
                        {
                            printf("Infeasible hyper-rectangle contains children:\n");
                            printf("   min: %5.3f  %5.3f  %5.3f  %5.3f\n"
                                   "   max: %5.3f  %5.3f  %5.3f  %5.3f\n",
                                   cell[0][0], cell[0][1], cell[0][2], cell[0][3],
                                   cell[1][0], cell[1][1], cell[1][2], cell[1][3] );
                            return 0;
                        }

                        pqueue_insert( queue,
                                quadtree::PriorityNode<float,4>(
                                    dist,
                                    child_depth,
                                    pair.second,
                                    cell ) );
                    }

                    for( auto& pair : cpu_child_dist )
                    {
                        quadtree::HyperRect<float,4> cell = pqnode.hrect;
                        std::bitset<4> bits( pair.first );
                        quadtree::refine<float,0,4,4>( cell,median,bits );

                        if( std::abs(pair.second - child_dist[pair.first]) > 1e-4 )
                        {
                            printf("gpu does not agree:\n");
                            printf("   min: %5.3f  %5.3f  %5.3f  %5.3f\n"
                                   "   max: %5.3f  %5.3f  %5.3f  %5.3f\n",
                                   cell[0][0], cell[0][1], cell[0][2], cell[0][3],
                                   cell[1][0], cell[1][1], cell[1][2], cell[1][3] );
                            printf("   child: %d\n",pair.first);
                            printf("   gpu:   %f\n",child_dist[pair.first] );
                            printf("   cpu:   %f\n",pair.second);
                            printf("   all gpu:\n");
                            for( auto& pair : cpu_child_dist )
                                printf( "     %2d : %6.4f   %6.4f   %6.4f \n",
                                        pair.first,
                                        pair.second,
                                        child_dist[pair.first],
                                        std::abs(pair.second-
                                                child_dist[pair.first])
                                        );
                            return 0;
                        }
                    }
                }
            }

            if( result.size() < k )
            {
                printf("   %d => FAIL\n",i);
                for(int m=0; m < k; m++)
                    printf("%3d  ",result_truth[m]);
                printf("\n");
                for(int m=0; m < result.size(); m++)
                    printf("%3d  ",result[m].point);
                return 0;
            }

            // compare result
            result_hope.clear();
            for(int i=0; i < k; i++ )
                result_hope.push_back( result[i].point );

            std::sort( result_truth.begin(), result_truth.end() );
            std::sort( result_hope.begin(),  result_hope.end() );

            for(int j=0; j < k; j++)
            {
                if( result_hope[j] != result_truth[j] )
                {
                    printf("   %d => FAIL\n",i);
                    for(int m=0; m < k; m++)
                        printf("%3d  ",result_truth[m]);
                    printf("\n");
                    for(int m=0; m < k; m++)
                        printf("%3d  ",result_hope[m]);
                    return 0;
                }
            }

            clock_gettime(CLOCK_MONOTONIC,&sw_stop);
            if( (sw_stop-app_start).milliseconds() - lastPrint > printTimeout )
            {
                lastPrint += printTimeout;
                printf("         queries: %6.2f%%\r", 100.0*i/queries.size());
                std::cout.flush();
            }
        }
        printf("         queries: %6.2f%%\n", 100.0);
    }

    printf("   gpu time: %f\n",dist_gpu.milliseconds());
    printf("   cpu time: %f\n",dist_cpu.milliseconds());
    printf("   speedup:  %f\n", dist_cpu.milliseconds() / dist_gpu.milliseconds()  );
}





