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
using namespace euclidean;



int main( int argc, char** argv )
{
    utility::Timespec app_start,sw_start,sw_stop;

    std::vector<Point2>    points2;
    std::vector<Point2>    queries2;

    std::vector<Point3>    points3;
    std::vector<Point3>    queries3;

    std::vector<int> result_truth;
    std::vector<int> result_hope;

    Implementation<2>* rqt2 = impl_rqt2(points2,10);
    Implementation<3>* rqt3 = impl_rqt3(points3,10);

    int maxSamples = 0;
    for(int numSamples : numSamplesList )
        if( numSamples > maxSamples )
            maxSamples = numSamples;

    points2 .resize(maxSamples,Point2(0,0));
    queries2.resize(numQueries,Point2(0,0));
    points3 .resize(maxSamples,Point3(0,0,0));
    queries3.resize(numQueries,Point3(0,0,0));

    rqt2->allocate(maxSamples);
    rqt3->allocate(maxSamples);

    std::cout << "Generating point set\n";

    srand( time(NULL) );

    double lastPrint = 0;
    clock_gettime(CLOCK_MONOTONIC,&app_start);
    for( int i=0; i < points2.size(); i++ )
    {
        Point2& point = points2[i];
        point = Point2( ws * rand() / float(RAND_MAX),
                       ws * rand() / float(RAND_MAX) );

        clock_gettime(CLOCK_MONOTONIC,&sw_stop);
        if( (sw_stop-app_start).milliseconds() - lastPrint > printTimeout  )
        {
            lastPrint += printTimeout;
            printf("   R^2 : %6.2f%%\r",100.0*i/points2.size());
            std::cout.flush();
        }
    }
    printf("   R^2 : %6.2f%%\n",100.0);

    for( int i=0; i < points3.size(); i++ )
    {
        Point3& point = points3[i];
        point = Point3( ws * rand() / float(RAND_MAX),
                        ws * rand() / float(RAND_MAX),
                        ws * rand() / float(RAND_MAX) );

        clock_gettime(CLOCK_MONOTONIC,&sw_stop);
        if( (sw_stop-app_start).milliseconds() - lastPrint > printTimeout  )
        {
            lastPrint += printTimeout;
            printf("   R^3 : %6.2f%%\r",100.0*i/points2.size());
            std::cout.flush();
        }
    }
    printf("   R^3 : %6.2f%%\n",100.0);

    for( Point2& point : queries2 )
        point = Point2( ws*rand() / float(RAND_MAX),
                       ws*rand() / float(RAND_MAX) );
    for( Point3& point : queries3 )
        point = Point3( ws*rand() / float(RAND_MAX),
                        ws*rand() / float(RAND_MAX),
                        ws*rand() / float(RAND_MAX) );

    for( int iNumSamples = 1;
            iNumSamples < numSamplesList.size(); iNumSamples++ )
    {
        int numSamples = numSamplesList[iNumSamples];
        int oldKdSize  = numSamplesList[iNumSamples-1];
        std::cout << "numSamples: " << numSamples << "\n";

        for(int i=oldKdSize; i<numSamples; i++)
        {
            rqt2->insert(i,points2[i]);
            rqt3->insert(i,points3[i]);

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
        for( int i=0; i < numQueries; i++ )
        {
            const Point2& q2 = queries2[i];
            const Point3& q3 = queries3[i];

            rqt2->findNearest(q2);
            rqt3->findNearest(q3);

            clock_gettime(CLOCK_MONOTONIC,&sw_stop);
            if( (sw_stop-app_start).milliseconds() - lastPrint > printTimeout )
            {
                lastPrint += printTimeout;
                printf("         queries: %6.2f%%\r", 100.0*i/numQueries);
                std::cout.flush();
            }
        }
        printf("         queries: %6.2f%%\n", 100.0);
    }

    std::ofstream out("./profiles_euclidean.csv");

    std::string name = "rqt2";
    for( auto& kv : rqt2->profiles )
    {
        out   << name << ","
              << kv.first << ","
              << kv.second.runtime.milliseconds()/kv.second.numTrials
              << ","
              << kv.second.nodesTouched/double(kv.second.numTrials)
              << "\n";
    }

    name = "rqt3";
    for( auto& kv : rqt3->profiles )
    {
        out   << name << ","
              << kv.first << ","
              << kv.second.runtime.milliseconds()/kv.second.numTrials
              << ","
              << kv.second.nodesTouched/double(kv.second.numTrials)
              << "\n";
    }

    std::cout << "\nExperiment finished\n";
}





