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




int main( int argc, char** argv )
{
    utility::Timespec app_start,sw_start,sw_stop;

    std::vector<Point>    points;
    std::vector<Point>    queries;

    std::vector<int> result_truth;
    std::vector<int> result_hope;

    std::map< std::string, Implementation* > impls =
    {
//        {"cuda",   impl_cuda()            },
//        {"old kd", impl_old_kd()          },
        {"nkd",     impl_kd2(points,10)    },
//        {"kdx10",   impl_kd2(points,10)   },
//        {"kdx50",   impl_kd2(points,50)   },
//        {"skd2",    impl_skd(points)      },
        {"skd4",    impl_sqt(points,10,4) },
//        {"rqt",     impl_rqt(points,10)   },
        {"uskd",    impl_ukd(points,10)   },
//        {"ukdx50",  impl_ukd(points,50)   }
    };

    int maxSamples = 0;
    for(int numSamples : numSamplesList )
        if( numSamples > maxSamples )
            maxSamples = numSamples;

    points .resize(maxSamples,Point(0,0,0));
    queries.resize(numQueries,Point(0,0,0));

    for( auto& pair : impls )
        pair.second->allocate(maxSamples);


    std::cout << "Generating point set\n";

    //srand( time(NULL) );
    srand(0);

    double lastPrint = 0;
    clock_gettime(CLOCK_MONOTONIC,&app_start);
    for( int i=0; i < points.size(); i++ )
    {
        Point& point = points[i];
        point = Point( ws * rand() / float(RAND_MAX),
                       ws * rand() / float(RAND_MAX),
                       (2*(rand()/float(RAND_MAX))-1)*M_PI );

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
        point = Point( ws*rand() / float(RAND_MAX),
                       ws*rand() / float(RAND_MAX),
                       (2*(rand()/float(RAND_MAX))-1)*M_PI );

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

            impls["nkd"]->findNearest(q);
            impls["nkd"]->get_result(result_truth);
            std::sort( result_truth.begin(),
                       result_truth.end() );

            for( auto& pair : impls )
            {
                if( pair.first != "nkd" )
                {
                    pair.second->findNearest(q);
                    pair.second->get_result(result_hope);

                    std::sort( result_hope.begin(),
                    result_hope.end() );

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
                            printf("\n");
                            break;
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


    std::ofstream out("./profiles_r2s1.csv");

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





