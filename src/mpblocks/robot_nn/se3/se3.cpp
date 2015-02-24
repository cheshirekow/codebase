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

#include <Eigen/Dense>
#include <mpblocks/utility/Timespec.h>

#include "config.h"
#include "Profile.h"
#include "Implementation.h"
#include "impl/ctors.h"

const bool verify = false;
const std::vector<int>  numSamplesList =
{
    0,
    100,
    200,
    400,
    1000,
    2000,
    4000,
    10000,
    20000,
    40000,
    100000,
    200000,
    400000,
    800000,
    1000000,
//    2000000,
//    4000000,
//    8000000,
//    10000000,
};

using namespace mpblocks;
using namespace robot_nn;
using namespace se3;

void box_muller(float& oa, float& ob)
{
    float u1 = rand() / float(RAND_MAX);
    float u2 = rand() / float(RAND_MAX);

    float a = std::sqrt( float(-2)*std::log(u1) );
    float b = float(2*M_PI)*u2;

    oa = a*std::cos(b);
    ob = b*std::sin(b);
}

void uniform_se3( Point& point )
{
    Eigen::Vector4f q;
    box_muller( q[0], q[1] );
    box_muller( q[2], q[3] );
    q.normalize();

    point << ws * rand() / float(RAND_MAX),
             ws * rand() / float(RAND_MAX),
             ws * rand() / float(RAND_MAX),
             q[0], q[1], q[2], q[3];
}

int main( int argc, char** argv )
{
    utility::Timespec app_start,sw_start,sw_stop;

    std::vector<Point>    points;
    std::vector<Point>    queries;

    std::vector<int> result_truth;
    std::vector<int> result_hope;

    std::map< std::string, Implementation* > impls =
    {
//        {"gpu_bf",      impl_gpu_bf()             },
        {"cpu_bf",      impl_cpu_bf(points)       },
        {"cpu_rqt",     impl_cpu_rqt(points,10)   },
//        {"gpu_rqt",     impl_gpu_rqt(points,10)   },
        {"cpu_sqt",     impl_cpu_sqt(points,10,4) },
//        {"gpu_sqt",     impl_gpu_sqt(points,10,4) },
    };

    int maxSamples = 0;
    for(int numSamples : numSamplesList )
        if( numSamples > maxSamples )
            maxSamples = numSamples;

    points .resize( maxSamples,Point::Zero() );
    queries.resize( numQueries,Point::Zero() );

    for( auto& pair : impls )
        pair.second->allocate(maxSamples);


    std::cout << "Generating point set\n";
    srand( time(NULL) );
    //srand(0);

    double lastPrint = 0;
    clock_gettime(CLOCK_MONOTONIC,&app_start);
    for( int i=0; i < points.size(); i++ )
    {
        Point& point = points[i];
        uniform_se3(point);

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
        uniform_se3(point);

    for( int iNumSamples = 1;
            iNumSamples < numSamplesList.size(); iNumSamples++ )
    {
        int numSamples = numSamplesList[iNumSamples];
        int oldKdSize  = numSamplesList[iNumSamples-1];
        std::cout << "numSamples: " << numSamples << "\n";

        for(int i=oldKdSize; i<numSamples; i++)
        {
            for( auto& pair : impls )
                pair.second->insert(i,points[i]);

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

        // insert points into kd tree
        // insert points into my data structure
        for( int i=0; i < queries.size(); i++ )
        {
            const Point& q = queries[i];

            if( impls.find("cpu_bf") != impls.end() )
                impls["cpu_bf"]->findNearest(q);

            if(verify && impls.find("cpu_bf") != impls.end() )
            {
                impls["cpu_bf"]->get_result(result_truth);
                std::sort( result_truth.begin(), result_truth.end() );
            }

            for( auto& pair : impls )
            {
                if( pair.first != "cpu_bf" )
                {
                    pair.second->findNearest(q);

                    if(verify && impls.find("cpu_bf") != impls.end() )
                    {
                        pair.second->get_result(result_hope);
                        std::sort( result_hope.begin(), result_hope.end() );
                        for(int j=0; j < k; j++)
                        {
                            if( result_hope[j] != result_truth[j] )
                            {
                                printf( "%5s : %d => FAIL\n",pair.first.c_str(),i);

                                for(int m=0; m < k; m++)
                                    printf("%3d  ",result_truth[m]);
                                printf("\n");
                                for(int m=0; m < k; m++)
                                    printf("%3d  ",result_hope[m]);

                                return 0;
                            }
                        }
                    }
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


    std::ofstream out("./profiles_se3.csv");
    for( auto& pair : impls )
    {
        const std::string& name = pair.first;
        for( auto& kv : pair.second->profiles )
        {
            out   << name << ","
                  << kv.first << ","
                  << kv.second.runtime.milliseconds()/kv.second.numTrials
                  << ","
                  << kv.second.nodesTouched/double(kv.second.numTrials)
                  << "\n";
        }
    }

    std::cout << "\nExperiment finished\n";
}





