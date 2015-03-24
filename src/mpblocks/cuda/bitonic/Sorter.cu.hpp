/*
 *  \file   bitonicSort.h
 *
 *  \date   Sep 3, 2011
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief
 *
 *  Note: adapted from the NVIDIA SDK bitonicSort.cu which references
 *  http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm
 */



#ifndef GPURRTS_BITONIC_SORTER_CU_HPP
#define GPURRTS_BITONIC_SORTER_CU_HPP

#include <iostream>
#include <cassert>
#include <algorithm>
#include <boost/format.hpp>



namespace mpblocks {
namespace     cuda {
namespace  bitonic {


inline void print_kernel_info(
       const char* kernel, const cuda::FuncAttributes& attr )
{
    typedef boost::format fmt;

    std::cout
        << fmt("   %s:\n") % kernel
        << fmt("         local: %i\n") % attr.localSizeBytes
        << fmt("      register: %i\n") % attr.numRegs
        << fmt("        shared: %i\n") % attr.sharedSizeBytes
        << fmt("      constant: %i\n") % attr.constSizeBytes;
}




template <typename KeyType, typename ValueType>
Sorter<KeyType,ValueType>::Sorter(KeyType min, KeyType max):
    m_min(min),
    m_max(max)
{
    config();
}



template <typename KeyType, typename ValueType>
void Sorter<KeyType,ValueType>::config()
{
    int dev = cuda::getDevice();
    config(dev);
}

template <typename KeyType, typename ValueType>
void Sorter<KeyType,ValueType>::config(int dev)
{
    typedef boost::format fmt;
    cuda::DeviceProp devProps(dev);

//    std::cout << "In Sorter c'tor:\n"
//        << fmt("          Size of key type: %i bytes\n") % sizeof(KeyType)
//        << fmt("        Size of value type: %i bytes\n") % sizeof(ValueType)
//        << fmt("   Shared memory per block: %i\n" ) % devProps.sharedMemPerBlock
//        << fmt("       Registers per block: %i\n" ) % devProps.regsPerBlock;

    cuda::FuncAttributes attr;
//    std::cout << "Kernel Memory Usage:\n";

    uint_t  maxShared   = 0;
    uint_t  maxRegs     = 0;

    attr.getFrom( &sortShared<KeyType,ValueType> );
    maxShared   = std::max(maxShared, (uint_t)attr.sharedSizeBytes);
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);
//    print_kernel_info( "bitonicSortShared", attr );

    attr.getFrom( &sortSharedInc<KeyType,ValueType>);
    maxShared   = std::max(maxShared, (uint_t)attr.sharedSizeBytes);
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);
//    print_kernel_info( "bitonicSortSharedInc", attr );

    attr.getFrom( &mergeGlobal<KeyType,ValueType>);
    maxShared   = std::max(maxShared, (uint_t)attr.sharedSizeBytes);
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);
//    print_kernel_info( "bitonicMergeGlobal", attr );

    attr.getFrom( &mergeShared<KeyType,ValueType>);
    maxShared   = std::max(maxShared, (uint_t)attr.sharedSizeBytes);
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);
//    print_kernel_info( "bitonicMergeShared", attr );

    attr.getFrom( &bitonic::prepare<KeyType>);
    maxShared   = std::max(maxShared, (uint_t)attr.sharedSizeBytes);
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);
//    print_kernel_info( "bitonicPrepare", attr );

    // the maximum number of key/value pairs we can store in shared memory
    // is determined by first finding out how much shared memory is left in a
    // block after subtracting shared memory for parameter storage and the like
    // and then dividing that amount by the number of bytes consumed by a
    // single key/value pair
    uint_t  sharedLength_max    = ((uint_t)devProps.sharedMemPerBlock - maxShared)/
                                    ( sizeof(KeyType) + sizeof(ValueType) );

    // the maximum number of threads we can put into a block is given by the
    // number of registers on each SM divided by the number of registers that
    // are used by each thread in the kernel
    uint_t  threadCount_max     = (uint_t)devProps.regsPerBlock / maxRegs;

    // the number of key/value pairs we actually store must be a power of two,
    // so we take it to be the largest power of two not greater than the minimum
    // of the two constraints calculated above
    m_sharedLength = prevPow2( std::min(sharedLength_max, 2*threadCount_max) );
    m_nSM          = devProps.multiProcessorCount;

    attr.getFrom( &mergeGlobal<KeyType,ValueType>);
    m_threadsMerge = prevPow2(attr.maxThreadsPerBlock);

    attr.getFrom( &bitonic::prepare<KeyType> );
    m_threadsPrepare = attr.maxThreadsPerBlock;

//    std::cout
//       << fmt("          Shared Length: %i\n") % m_sharedLength
//       << fmt("        Multiprocessors: %i\n") %  m_nSM
//       << fmt("  Max Threads for merge: %i\n") %  attr.maxThreadsPerBlock
//       << fmt("          however using: %i\n") %  m_threadsMerge
//       << fmt("Max Threads for prepare: %i\n") % attr.maxThreadsPerBlock
//       << fmt("          however using: %i\n") %  m_threadsPrepare;
}




template <typename KeyType, typename ValueType>
uint_t Sorter<KeyType,ValueType>::sort(
                    KeyType  *d_DstKey,
                    ValueType   *d_DstVal,
                    KeyType     *d_SrcKey,
                    ValueType   *d_SrcVal,
                    uint_t      arrayLength,
                    Direction   dir
                    )
{
    // if the array is not a power of two, then find the next higher power
    // of two for the array length and prepare everything from the end of
    // the array up to that power of two length (by prepare, I mean set them
    // all to the max or min value, i.e. zero or inf for unsigned integers).
    uint_t  actualArrayLength   = arrayLength;
            arrayLength         = nextPow2(arrayLength);
    uint_t  prepareThreads      = arrayLength - actualArrayLength;

    if(prepareThreads)
        this->prepare(d_SrcKey+actualArrayLength, prepareThreads, dir);

    uint_t threadCount =
            bitonic::sort<KeyType,ValueType>(
                                            d_DstKey, d_DstVal,
                                            d_SrcKey, d_SrcVal,
                                            arrayLength, m_sharedLength, dir,
                                            m_threadsMerge);

    // will throw an exception if the kernel call failed
    cuda::checkLastError("bitonic::sort");

    return threadCount;
}




template <typename KeyType, typename ValueType>
void Sorter<KeyType,ValueType>::prepare(  KeyType* d_SrcKey,
                            uint_t arrayLength,
                            Direction dir)
{
    uint_t  blockCount, threadCount;

    if(m_threadsPrepare < arrayLength)
    {
        blockCount  = intDivideRoundUp(arrayLength , m_threadsPrepare);
        threadCount = intDivideRoundUp(arrayLength, blockCount);
    }
    else
    {
        blockCount  = 1;
        threadCount = arrayLength;
    }

    if(dir == Ascending)
    {
//        std::cout << "Preparing sort by padding with: " << m_max << "\n";
        bitonic::prepare<KeyType><<<blockCount,threadCount>>>(d_SrcKey,
                            m_max,
                            arrayLength);
    }
    else
    {
//        std::cout << "Preparing sort by padding with: " << m_min << "\n";
        bitonic::prepare<KeyType><<<blockCount,threadCount>>>(d_SrcKey,
                            m_min,
                            arrayLength);
    }

    cuda::checkLastError("bitonic::prepare");
}






































































template <typename KeyType>
Sorter<KeyType,void>::Sorter(KeyType min, KeyType max):
    m_min(min),
    m_max(max)
{
    config();
}



template <typename KeyType>
void Sorter<KeyType,void>::config()
{
    int dev = cuda::getDevice();
    config(dev);
}

template <typename KeyType>
void Sorter<KeyType,void>::config(int dev)
{
    typedef boost::format fmt;
    cuda::DeviceProp devProps(dev);

    std::cout << "In Sorter c'tor:\n"
        << fmt("          Size of key type: %i bytes\n") % sizeof(KeyType)
        << fmt("        Size of value type: %i bytes\n") % 0
        << fmt("   Shared memory per block: %i\n" ) % devProps.sharedMemPerBlock
        << fmt("       Registers per block: %i\n" ) % devProps.regsPerBlock;

    cuda::FuncAttributes attr;
    std::cout << "Kernel Memory Usage:\n";

    uint_t  maxShared   = 0;
    uint_t  maxRegs     = 0;

    attr.getFrom( &sortShared<KeyType> );
    maxShared   = std::max(maxShared, (uint_t)attr.sharedSizeBytes);
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);
    print_kernel_info( "bitonicSortShared", attr );

    attr.getFrom( &sortSharedInc<KeyType>);
    maxShared   = std::max(maxShared, (uint_t)attr.sharedSizeBytes);
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);
    print_kernel_info( "bitonicSortSharedInc", attr );

    attr.getFrom( &mergeGlobal<KeyType>);
    maxShared   = std::max(maxShared, (uint_t)attr.sharedSizeBytes);
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);
    print_kernel_info( "bitonicMergeGlobal", attr );

    attr.getFrom( &mergeShared<KeyType>);
    maxShared   = std::max(maxShared, (uint_t)attr.sharedSizeBytes);
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);
    print_kernel_info( "bitonicMergeShared", attr );

    attr.getFrom( &bitonic::prepare<KeyType>);
    maxShared   = std::max(maxShared, (uint_t)attr.sharedSizeBytes);
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);
    print_kernel_info( "bitonicPrepare", attr );

    // the maximum number of key/value pairs we can store in shared memory
    // is determined by first finding out how much shared memory is left in a
    // block after subtracting shared memory for parameter storage and the like
    // and then dividing that amount by the number of bytes consumed by a
    // single key/value pair
    uint_t  sharedLength_max    = ((uint_t)devProps.sharedMemPerBlock - maxShared)/
                                    ( sizeof(KeyType)  );

    // the maximum number of threads we can put into a block is given by the
    // number of registers on each SM divided by the number of registers that
    // are used by each thread in the kernel
    uint_t  threadCount_max     = (uint_t)devProps.regsPerBlock / maxRegs;

    // the number of key/value pairs we actually store must be a power of two,
    // so we take it to be the largest power of two not greater than the minimum
    // of the two constraints calculated above
    m_sharedLength = prevPow2( std::min(sharedLength_max, 2*threadCount_max) );
    m_nSM = devProps.multiProcessorCount;

    attr.getFrom(&mergeGlobal<KeyType>);
    m_threadsMerge = prevPow2(attr.maxThreadsPerBlock);

    attr.getFrom(&bitonic::prepare<KeyType>);
    m_threadsPrepare = attr.maxThreadsPerBlock;

    std::cout
       << fmt("          Shared Length: %i\n") % m_sharedLength
       << fmt("        Multiprocessors: %i\n") %  m_nSM
       << fmt("  Max Threads for merge: %i\n") %  attr.maxThreadsPerBlock
       << fmt("          however using: %i\n") %  m_threadsMerge
       << fmt("Max Threads for prepare: %i\n") % attr.maxThreadsPerBlock
       << fmt("          however using: %i\n") %  m_threadsPrepare;
}




template <typename KeyType>
uint_t Sorter<KeyType,void>::sort(
                    KeyType  *d_DstKey,
                    KeyType     *d_SrcKey,
                    uint_t      arrayLength,
                    Direction   dir
                    )
{
    // if the array is not a power of two, then find the next higher power
    // of two for the array length and prepare everything from the end of
    // the array up to that power of two length (by prepare, I mean set them
    // all to the max or min value, i.e. zero or inf for unsigned integers).
    uint_t  actualArrayLength   = arrayLength;
            arrayLength         = nextPow2(arrayLength);
    uint_t  prepareThreads      = arrayLength - actualArrayLength;

    if(prepareThreads)
    {
        this->prepare(d_SrcKey+actualArrayLength, prepareThreads, dir);
    }


    uint_t threadCount =
            bitonic::sort<KeyType>( d_DstKey,
                                    d_SrcKey,
                                    arrayLength, m_sharedLength, dir,
                                    m_threadsMerge );

    // will throw an exception if the kernel call failed
    cuda::checkLastError("bitonic::sort");

    return threadCount;
}




template <typename KeyType>
void Sorter<KeyType,void>::prepare(  KeyType* d_SrcKey,
                            uint_t arrayLength,
                            Direction dir)
{
    uint_t  blockCount, threadCount;

    if(m_threadsPrepare < arrayLength)
    {
        blockCount  = intDivideRoundUp(arrayLength , m_threadsPrepare);
        threadCount = intDivideRoundUp(arrayLength, blockCount);
    }
    else
    {
        blockCount  = 1;
        threadCount = arrayLength;
    }

    if(dir == Ascending)
    {
        bitonic::prepare<KeyType><<<blockCount,threadCount>>>(d_SrcKey,
                            m_max,
                            arrayLength);
    }
    else
    {
        bitonic::prepare<KeyType><<<blockCount,threadCount>>>(d_SrcKey,
                            m_min,
                            arrayLength);
    }

    cuda::checkLastError("bitonic::prepare");
}




} // namespace bitonic
} // namespace cuda
} // namespace mpblocks



#endif
