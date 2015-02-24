/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implemenets bitonic sort and odd-even merge sort, algorithms
 * belonging to the class of sorting networks. 
 * While generally subefficient on large sequences 
 * compared to algorithms with better asymptotic algorithmic complexity
 * (i.e. merge sort or radix sort), may be the algorithms of choice for sorting
 * batches of short- or mid-sized arrays.
 * Refer to the excellent tutorial by H. W. Lang:
 * http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm
 * 
 * Victor Podlozhnyuk, 07/09/2009
 */

#include <cstdlib>
#include <limits>
#include <iostream>
#include <boost/format.hpp>

#include <mpblocks/utility/Timespec.h>
#include <mpblocks/cuda.hpp>
#include <mpblocks/cuda/bitonic.h>


using namespace mpblocks;
using namespace mpblocks::utility;
using namespace mpblocks::cuda;
using namespace mpblocks::cuda::bitonic;

typedef boost::format fmt;
typedef unsigned int  uint;

////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    Timespec start, end;

    std::cout << "Starting up CUDA context\n";
    cudaSetDevice(0);

    int dev;
    cudaGetDevice(&dev);
    std::cout << "Current device: " << dev << "\n";
    
    unsigned int *h_InputKey, *h_InputVal, *h_OutputKeyGPU, *h_OutputValGPU;
    unsigned int *d_InputKey, *d_InputVal, *d_OutputKey,    *d_OutputVal;

    const size_t                N           = twoPow(14);
    const bitonic::Direction    DIR         = Descending;
    const size_t                numValues   = twoPow(16);

    std::cout << "Allocating and initializing host arrays...\n";

    h_InputKey     = (uint *)::malloc(N * sizeof(uint));
    h_InputVal     = (uint *)::malloc(N * sizeof(uint));
    h_OutputKeyGPU = (uint *)::malloc(N * sizeof(uint));
    h_OutputValGPU = (uint *)::malloc(N * sizeof(uint));

    std::cout << "Allocating and initializing CUDA arrays...\n";
    d_InputKey  = mallocT<uint>( N );
    d_InputVal  = mallocT<uint>( N );
    d_OutputKey = mallocT<uint>( N );
    d_OutputVal = mallocT<uint>( N );

    std::cout << "Creating bitonic sort object\n";
    bitonic::Sorter<unsigned int, unsigned int> sorter(
                            std::numeric_limits<unsigned int>::min(),
                            std::numeric_limits<unsigned int>::max() );
    bitonic::Sorter<unsigned int> keySorter(
                            std::numeric_limits<unsigned int>::min(),
                            std::numeric_limits<unsigned int>::max() );

    for(uint arrayLength = 64; arrayLength <= N; arrayLength *= 2)
    {
        std::cout << fmt("Testing array length %u ...\n") % arrayLength;
        std::cout << "   generating random numbers\n";
        srand(2001);
        for(size_t i = 0; i < arrayLength; i++){
            h_InputKey[i] = rand() % numValues;
            h_InputVal[i] = i;
        }

        std::cout << "   copying from host to device\n";
        memcpyT( d_InputKey, h_InputKey, arrayLength, cudaMemcpyHostToDevice );
        memcpyT( d_InputVal, h_InputVal, arrayLength, cudaMemcpyHostToDevice );
        deviceSynchronize();

        //timer.reset();
        clock_gettime( CLOCK_MONOTONIC, &start );
        uint threadCount = sorter.sort( d_OutputKey,
                                        d_OutputVal,
                                        d_InputKey,
                                        d_InputVal,
                                        arrayLength,
                                        DIR             );

        deviceSynchronize();
        clock_gettime( CLOCK_MONOTONIC, &end );


        double ms = (end-start).milliseconds();
        std::cout
            << fmt( "      Time: %f ms\n"        ) % ms
            << fmt( "Throughput: %f Elements/s\n") % (1000.0*arrayLength/ms)
            << fmt( " Workgroup: %i threads\n"   ) % threadCount;
        std::cout << "Validating the results...\n"
                  << "   ... reading back GPU results\n";

        memcpyT(h_OutputKeyGPU, d_OutputKey, arrayLength, cudaMemcpyDeviceToHost );
        memcpyT(h_OutputValGPU, d_OutputVal, arrayLength, cudaMemcpyDeviceToHost );

        int sortFlag  = 0;
        int valueFlag = 0;
        for(size_t i=0; i < arrayLength-1; i++ )
        {
            if( h_OutputKeyGPU[i] < h_OutputKeyGPU[i+1] )
                sortFlag = i;

            if(h_OutputValGPU[i] < arrayLength)
            {
                if( h_InputKey[h_OutputValGPU[i]] != h_OutputKeyGPU[i] )
                    valueFlag = i;
            }
            else
                valueFlag = 0;

            if(valueFlag || sortFlag)
                break;
        }

        std::cout
            << fmt("Sort verify:  %s (%i)\n") % (sortFlag  ? "FAIL" : "OK") % sortFlag
            << fmt("Value verify: %s (%i)\n") % (valueFlag ? "FAIL" : "OK") % valueFlag
            << "\n";
    }


    for(uint arrayLength = 64; arrayLength <= 128; arrayLength += 32)
    {
        std::cout << fmt( "Testing array length %u ...\n" ) % arrayLength
                  << "   generating random numbers\n";
        srand(2001);
        for(size_t i = 0; i < arrayLength; i++){
            h_InputKey[i] = rand() % numValues;
            h_InputVal[i] = i;
        }

        std::cout << "   copying from host to device\n";
        memcpyT( d_InputKey, h_InputKey, arrayLength, cudaMemcpyHostToDevice );
        memcpyT( d_InputVal, h_InputVal, arrayLength, cudaMemcpyHostToDevice );

        deviceSynchronize();

        clock_gettime( CLOCK_MONOTONIC, &start );
        uint threadCount = sorter.sort( d_OutputKey,
                                        d_OutputVal,
                                        d_InputKey,
                                        d_InputVal,
                                        arrayLength,
                                        DIR             );
        deviceSynchronize();
        clock_gettime( CLOCK_MONOTONIC, &end );

        double ms = (end-start).milliseconds();
        std::cout
            << fmt( "      Time: %f ms\n"        ) % ms
            << fmt( "Throughput: %f Elements/s\n") % (1000.0*arrayLength/ms)
            << fmt( " Workgroup: %i threads\n"   ) % threadCount;
        std::cout << "Validating the results...\n"
                  << "   ... reading back GPU results\n";

        memcpyT(h_OutputKeyGPU, d_OutputKey, arrayLength,cudaMemcpyDeviceToHost);
        memcpyT(h_OutputValGPU, d_OutputVal, arrayLength,cudaMemcpyDeviceToHost);

        int sortFlag  = 0;
        int valueFlag = 0;
        for(size_t i=0; i < arrayLength-1; i++ )
        {
            if( h_OutputKeyGPU[i] < h_OutputKeyGPU[i+1] )
                sortFlag = i;

            if(h_OutputValGPU[i] < arrayLength)
            {
                if( h_InputKey[h_OutputValGPU[i]] != h_OutputKeyGPU[i] )
                    valueFlag = i;
            }
            else
                valueFlag = 0;

            if(valueFlag || sortFlag)
                break;
        }

        std::cout
            << fmt("Sort verify:  %s (%i)\n") % (sortFlag  ? "FAIL" : "OK") % sortFlag
            << fmt("Value verify: %s (%i)\n") % (valueFlag ? "FAIL" : "OK") % valueFlag
            << "\n";
    }



    for(uint arrayLength = 64; arrayLength <= 128; arrayLength += 32)
    {
        std::cout << fmt( "Testing array length %u ...\n" ) % arrayLength
                  << "   generating random numbers\n";
        srand(2001);
        for(size_t i = 0; i < arrayLength; i++){
            h_InputKey[i] = rand() % numValues;
        }

        std::cout << "   copying from host to device\n";
        memcpyT( d_InputKey, h_InputKey, arrayLength, cudaMemcpyHostToDevice );
        deviceSynchronize();

        clock_gettime( CLOCK_MONOTONIC, &start );
        uint threadCount = keySorter.sort( d_OutputKey,
                                        d_InputKey,
                                        arrayLength,
                                        DIR             );
        deviceSynchronize();
        clock_gettime( CLOCK_MONOTONIC, &end );


        double ms = (end-start).milliseconds();
        std::cout
            << fmt( "      Time: %f ms\n"        ) % ms
            << fmt( "Throughput: %f Elements/s\n") % (1000.0*arrayLength/ms)
            << fmt( " Workgroup: %i threads\n"   ) % threadCount;
        std::cout << "Validating the results...\n"
                  << "   ... reading back GPU results\n";

        memcpyT(h_OutputKeyGPU, d_OutputKey, arrayLength, cudaMemcpyDeviceToHost);

        int sortFlag  = 0;
        for(size_t i=0; i < arrayLength-1; i++ )
        {
            if( h_OutputKeyGPU[i] < h_OutputKeyGPU[i+1] )
                sortFlag = i;

            if(sortFlag)
                break;
        }

        std::cout
            << fmt("Sort verify:  %s (%i)\n") % (sortFlag  ? "FAIL" : "OK") % sortFlag
            << "\n";
    }

    std::cout << "Shutting down...\n";
    cuda::free(d_OutputVal);
    cuda::free(d_OutputKey);
    cuda::free(d_InputVal);
    cuda::free(d_InputKey);
    ::free(h_OutputValGPU);
    ::free(h_OutputKeyGPU);
    ::free(h_InputVal);
    ::free(h_InputKey);
}
