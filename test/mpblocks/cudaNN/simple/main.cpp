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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_nn/test/simple/main.cpp
 *
 *  @date   Oct 7, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */


#include <vector>
#include <map>
#include <iostream>
#include <cstdio>
#include <mpblocks/cuda.h>
#include <mpblocks/cudaNN/PointSet.h>
#include <Eigen/Dense>

using namespace mpblocks;
using namespace mpblocks::cudaNN;

struct Point
{
    float data[7];
    float& operator[](int i){ return data[i]; }
};

float pseudo_distance(
                float w,
                const Eigen::Vector3f& t0,
                const Eigen::Vector4f& q0,
                const Eigen::Vector3f& t1,
                const Eigen::Vector4f& q1 )
{
    float dq = q0.dot(q1);
    return (t1-t0).squaredNorm() + w * (1-dq*dq);
}

float distance(
                float w,
                const Eigen::Vector3f& t0,
                const Eigen::Vector4f& q0,
                const Eigen::Vector3f& t1,
                const Eigen::Vector4f& q1 )
{
    float dq  = q0.dot(q1);
    float arg = 2*dq*dq - 1;
    arg = fmaxf(-0.999999999999999999999999999f,
          fminf(arg, 0.9999999999999999999999999f));
    return (t1-t0).norm() + w* acosf( arg );
}

float distance(
                float w,
                const Point& A,
                const Point& B)
{
    Eigen::Matrix<float,3,1> t0( A.data );
    Eigen::Matrix<float,4,1> q0( A.data +3 );
    Eigen::Matrix<float,3,1> t1( B.data );
    Eigen::Matrix<float,4,1> q1( B.data +3 );
    return distance(w,t0,q0,t1,q1);
}




int main(int argc, char** argv)
{
    cuda::setDevice(0);

    int nPoints = 1000;
    int nTrials = 100;
    int kNN     = 10;
    float w     = 1.0f;

    PointSet<float,7> ps7(nPoints);
    ResultBlock<float,2> result(kNN);

    std::vector<Point> pts(nPoints);
    for(int i=0; i < pts.size(); i++)
    {
        Point& point = pts[i];
        for(int j=0; j < 3; j++)
            point[j] = rand() / (double)RAND_MAX;
        Eigen::Quaternionf q(
                rand() / (double)RAND_MAX,
                rand() / (double)RAND_MAX,
                rand() / (double)RAND_MAX,
                rand() / (double)RAND_MAX );
        q.normalize();
        point[3] = q.w();
        point[4] = q.x();
        point[5] = q.y();
        point[6] = q.z();

        ps7.insert(point.data);
    }


    for(int iTrial = 0; iTrial < nTrials; iTrial++)
    {
        Point point;
        for(int j=0; j < 3; j++)
            point[j] = rand() / (double)RAND_MAX;
        Eigen::Quaternionf q(
                rand() / (double)RAND_MAX,
                rand() / (double)RAND_MAX,
                rand() / (double)RAND_MAX,
                rand() / (double)RAND_MAX );
        q.normalize();
        point[3] = q.w();
        point[4] = q.x();
        point[5] = q.y();
        point[6] = q.z();

        int t = rand() % pts.size();
//        ps7.nearest( EUCLIDEAN, point.data, result );
        ps7.nearest( EUCLIDEAN, pts[t].data, result );

//        Eigen::Matrix<float,7,1> x0( point.data );
//        Eigen::Matrix<float,3,1> t0( point.data );
//        Eigen::Matrix<float,4,1> q0( point.data +3 );
        Eigen::Matrix<float,7,1> x0( pts[t].data );
        Eigen::Matrix<float,3,1> t0( pts[t].data );
        Eigen::Matrix<float,4,1> q0( pts[t].data +3 );

        std::multimap<float,float> euclidean_sorted;
        std::multimap<float,float> se3_sorted;
        for( int i=0; i < pts.size(); i++)
        {
            Eigen::Matrix<float,7,1> xi( pts[i].data );
            Eigen::Matrix<float,3,1> ti( pts[i].data );
            Eigen::Matrix<float,4,1> qi( pts[i].data +3 );
            euclidean_sorted.insert(
                    std::pair<float,float>( (x0-xi).squaredNorm(), i ) );
            se3_sorted.insert(
                    std::pair<float,float>( distance(w,t0,q0,ti,qi), i ) );
        }

        std::vector< std::pair<float,float> > cpu_result;
        cpu_result.reserve(kNN);
        cpu_result.insert( cpu_result.end(),
                            euclidean_sorted.begin(), euclidean_sorted.end() );
        bool euclideanOK = true;
        for(int i=0; i < kNN; i++)
        {
            if( cpu_result[i].second != result(1,i) )
                euclideanOK = false;
        }

//        ps7.nearest( SE3Tag(w), point.data, result );
        ps7.nearest( SE3Tag(w), pts[t].data, result );

        cpu_result.clear();
        cpu_result.reserve(kNN);
        cpu_result.insert( cpu_result.end(),
                            se3_sorted.begin(), se3_sorted.end() );
        bool se3OK = true;
        int idxT = -1;
        for(int i=0; i < kNN; i++)
        {
            if( cpu_result[i].second != result(1,i) )
                se3OK = false;
            if( cpu_result[i].second == t )
                idxT = i;
        }

        std::cout << iTrial << " : "
                    << "euclidean " << (euclideanOK ? "OK   " : "FAIL ")
                    << "se3  "      << (se3OK       ? "OK   " : "FAIL ")
                    << "idx t  "    << idxT
                    << "\n";
//        std::cout << "   cpu: " ;
//        for( int i=0; i < kNN; i++ )
//            printf("%3.0f  ",cpu_result[i].second);
//        std::cout << "\n   gpu: ";
//        for( int i=0; i < kNN; i++ )
//            printf("%3.0f  ",result(1,i));
//        std::cout << "\n";
//        if( !se3OK )
//        {
//            std::cout << "      q: ";
//            for(int i=0; i < 7; i++)
//                printf("%6.3f   ",pts[t][i]);
//            std::cout << "\n  d_cpu[0]: " << distance( w, pts[t], pts[cpu_result[0].second] )
//                      << "\n  d_gpu[0]: " << distance( w, pts[t], pts[result(1,0)] )
//                      << "\n";
//        }

//        std::cout << "\ngpu:   ";
//        for(int i=0; i < kNN; i++)
//            std::cout << result(0,i) << "  ";
//        std::cout << "\n";
//        for(int i=0; i < kNN; i++)
//            std::cout << result(1,i) << "  ";
//        std::cout << "\ncpu:   ";
//        for(int i=0; i < kNN; i++)
//            std::cout << cpu_result[i].first << "  ";
//        std::cout << "\n:   ";
//        for(int i=0; i < kNN; i++)
//            std::cout << cpu_result[i].second << "  ";
    }





    return 0;
}







