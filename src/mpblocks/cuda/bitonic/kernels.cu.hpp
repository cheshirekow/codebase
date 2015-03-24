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



#ifndef MPBLOCKS_CUDA_BITONIC_KERNELS_CU_HPP
#define MPBLOCKS_CUDA_BITONIC_KERNELS_CU_HPP

#include <cassert>
#include <algorithm>





// Map to single instructions on G8x / G9x / G100
// note: check hardware and see if we should change this, __umul24 is a single
// instruction multiply with 24 bit precision,
// note: it's only used to multiply block index times block dimension to get
// the block offset of a thread index, so it should be fine unless we need
// more than 2^24 = 16,777,216 threads
#define UMUL(a, b) __umul24((a), (b))
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )



namespace mpblocks {
namespace     cuda {
namespace  bitonic {


typedef unsigned int uint_t;


/// implements a "comparator": compares to keys and swaps them if they are not
/// in the desired order
/**
 *  there is no validation of this fact but it is required that the less-than
 *  operator (<) is defined for \p KeyType
 *
 *  \ingroup    bitonic
 */
template <typename KeyType, typename ValueType>
__device__ inline void compareSwap(
    KeyType&        keyA,   ///< the first key to compare
    ValueType&      valA,   ///< value associated with the first key
    KeyType&        keyB,   ///< the second key to compare
    ValueType&      valB,   ///< value associted with the second key
    Direction       dir     ///< the direction to sort, 1=ascending
)
{
    KeyType     tempKey;
    ValueType   tempValue;
    if( (keyA > keyB) == dir )
    {
        tempKey     = keyA; keyA = keyB; keyB = tempKey;
        tempValue   = valA; valA = valB; valB = tempValue;
    }
}




/// compares to keys and swaps them if they are not in the desired order
/**
 *  there is no validation of this fact but it is required that the less-than
 *  operator (<) is defined for \p KeyType
 *
 *  \ingroup    bitonic
 */
template <typename KeyType>
__device__ inline void compareSwap(
    KeyType&        keyA,   ///< the first key to compare
    KeyType&        keyB,   ///< the second key to compare
    Direction       dir     ///< the direction to sort, 1=ascending
)
{
    KeyType     tempKey;
    if( (keyA > keyB) == dir )
    {
        tempKey     = keyA; keyA = keyB; keyB = tempKey;
    }
}




/// single kernel (unified) bitonic sort
/**
 *  If the entire array to be sorted fits in shared memory, then we can perform
 *  the entire operation with only one kernel call (that's this kernel). If
 *  the entire array does not fit in shared memory then some of the
 *  comparator networks will have a stride large enough to cross block
 *  boundaries, so we have to divide and conquor (see the other kernels for this
 *  method)
 *
 *  \ingroup    bitonic
 */
template <typename KeyType, typename ValueType>
__global__ void sortShared(
    KeyType    *d_DstKey,
    ValueType  *d_DstVal,
    KeyType    *d_SrcKey,
    ValueType  *d_SrcVal,
    uint_t     arrayLength,
    Direction  dir
)
{
    //Shared memory storage for the shared vectors, in the kernel call this
    //array is allocated to have
    //arrayLength*(sizeof(KeyType)+sizeof(ValueType)) bytes
    extern __shared__ unsigned int array[];

    // we have to generate pointers into the shared memory block for our actual
    // shared array storage
    KeyType*     s_key = (KeyType*)array;
    ValueType*   s_val = (ValueType*)&s_key[arrayLength];

    //calculate offset for this thread from the start of the array
    d_SrcKey += blockIdx.x * arrayLength + threadIdx.x;
    d_SrcVal += blockIdx.x * arrayLength + threadIdx.x;
    d_DstKey += blockIdx.x * arrayLength + threadIdx.x;
    d_DstVal += blockIdx.x * arrayLength + threadIdx.x;

    // copy this threads data into shared memory
    s_key[threadIdx.x +                 0] = d_SrcKey[                0];
    s_val[threadIdx.x +                 0] = d_SrcVal[                0];
    s_key[threadIdx.x + (arrayLength / 2)] = d_SrcKey[(arrayLength / 2)];
    s_val[threadIdx.x + (arrayLength / 2)] = d_SrcVal[(arrayLength / 2)];

    for(uint_t size = 2; size < arrayLength; size <<= 1)
    {
        //Bitonic merge
        Direction ddd = (Direction)(
                ((unsigned int)dir) ^ ( (threadIdx.x & (size / 2)) != 0 ));

        for(uint_t stride = size / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            compareSwap(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                ddd
            );
        }
    }

    //ddd == dir for the last bitonic merge step
    for(uint_t stride = arrayLength / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        compareSwap(
            s_key[pos +      0], s_val[pos +      0],
            s_key[pos + stride], s_val[pos + stride],
            dir
        );
    }

    __syncthreads();
    d_DstKey[                0] = s_key[threadIdx.x +                 0];
    d_DstVal[                0] = s_val[threadIdx.x +                 0];
    d_DstKey[(arrayLength / 2)] = s_key[threadIdx.x + (arrayLength / 2)];
    d_DstVal[(arrayLength / 2)] = s_val[threadIdx.x + (arrayLength / 2)];
}




/// single kernel (unified) bitonic sort
/**
 *  If the entire array to be sorted fits in shared memory, then we can perform
 *  the entire operation with only one kernel call (that's this kernel). If
 *  the entire array does not fit in shared memory then some of the
 *  comparator networks will have a stride large enough to cross block
 *  boundaries, so we have to divide and conquor (see the other kernels for this
 *  method)
 *
 *  \ingroup    bitonic
 */
template <typename KeyType>
__global__ void sortShared(
    KeyType     *d_DstKey,
    KeyType     *d_SrcKey,
    uint_t      arrayLength,
    Direction   dir
)
{
    //Shared memory storage for the shared vectors, in the kernel call this
    //array is allocated to have
    //arrayLength*(sizeof(KeyType)+sizeof(ValueType)) bytes
    extern __shared__ unsigned int array[];

    // we have to generate pointers into the shared memory block for our actual
    // shared array storage
    KeyType*     s_key = (KeyType*)array;

    //calculate offset for this thread from the start of the array
    d_SrcKey += blockIdx.x * arrayLength + threadIdx.x;
    d_DstKey += blockIdx.x * arrayLength + threadIdx.x;

    // copyt this threads data into shared memory
    s_key[threadIdx.x +                 0] = d_SrcKey[                0];
    s_key[threadIdx.x + (arrayLength / 2)] = d_SrcKey[(arrayLength / 2)];

    for(uint_t size = 2; size < arrayLength; size <<= 1)
    {
        //Bitonic merge
        Direction ddd = (Direction)(
                ((unsigned int)dir) ^ ( (threadIdx.x & (size / 2)) != 0 ));

        for(uint_t stride = size / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            compareSwap(
                s_key[pos +      0],
                s_key[pos + stride],
                ddd
            );
        }
    }

    //ddd == dir for the last bitonic merge step
    for(uint_t stride = arrayLength / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        compareSwap(
            s_key[pos +      0],
            s_key[pos + stride],
            dir
        );
    }

    __syncthreads();
    d_DstKey[                0] = s_key[threadIdx.x +                 0];
    d_DstKey[(arrayLength / 2)] = s_key[threadIdx.x + (arrayLength / 2)];
}


















/// bottom level of the bitonic sort
/**
 *  Since this kernel works in shared memory we'd like to use it as much as
 *  possible, so what we do is divide up the entire array that we want to sort
 *  and only sort it in sections that are small enough to fit in shared
 *  memory. However, we sort every other block in a different direction.
 *  As a result, each pair of results forms a bitonic series. We can then
 *  efficiently merge each pair of blocks into a sorted series, which we
 *  continue doing until the entire array is sorted.
 *
 *  Note: the next stage (Bitonic merge) accepts both ascending | descending
 *  and descending | ascending bitonic series
 *
 *  \ingroup    bitonic
 */
template <typename KeyType, typename ValueType>
__global__ void sortSharedInc(
    KeyType     *d_DstKey,
    ValueType   *d_DstVal,
    KeyType     *d_SrcKey,
    ValueType   *d_SrcVal,
    uint_t      sharedLength
){
    //Shared memory storage for the shared vectors, in the kernel call this
    //array is allocated to have
    //sharedLength*(sizeof(KeyType)+sizeof(ValueType)) bytes
    extern __shared__ unsigned int array[];

    // we have to generate pointers into the shared memory block for our actual
    // shared array storage
    KeyType*     s_key = (KeyType*)array;
    ValueType*   s_val = (ValueType*)&s_key[sharedLength];

    // calculate this threads offset to the data
    d_SrcKey += blockIdx.x * sharedLength + threadIdx.x;
    d_SrcVal += blockIdx.x * sharedLength + threadIdx.x;
    d_DstKey += blockIdx.x * sharedLength + threadIdx.x;
    d_DstVal += blockIdx.x * sharedLength + threadIdx.x;

    // copy this threads data into shared memory
    s_key[threadIdx.x +                  0] = d_SrcKey[                 0];
    s_val[threadIdx.x +                  0] = d_SrcVal[                 0];
    s_key[threadIdx.x + (sharedLength / 2)] = d_SrcKey[(sharedLength / 2)];
    s_val[threadIdx.x + (sharedLength / 2)] = d_SrcVal[(sharedLength / 2)];


    // results in a tritonic series that can be split into two bitonic
    // serieses, i.e. steps a-e
    //
    // take a look at the comparator network drawing for the bitonic sort, this
    // first part is the first half of the network, we start by comparing only
    // pairs, (size=2), then we compaire quads (size=4)
    for(uint_t size = 2; size < sharedLength; size <<= 1)
    {
        // recall that size is always a power of two, so only one bit is
        // set to 1, so we shift that bit to the right by 1 and ask if that
        // new bit location is set in the thread id. If it is then the
        // direction is 1 otherwise it's 0
        Direction ddd = (Direction)( (threadIdx.x & (size / 2)) != 0 );

        // the stride is the distance (in indices) between the two keys being
        // compared.
        for(uint_t stride = size / 2; stride > 0; stride >>= 1)
        {
            // sync threads so that reads are not stale from the location of
            // another threads writes
            __syncthreads();

            // this math is really wonky, but I'm guessing that it resolves to
            // be the left side of the comparator network, i.e. when size = 2
            // then it does (x,y)->a, when size = 4 then it does a->(b,c) and
            // (b,c)->d, if the length of the example where bigger, this would
            // do more, but in the example this ends at e
            uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

            // perform the comparison/swap
            compareSwap(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                ddd
            );
        }
    }


    //Odd / even arrays of sharedLength elements
    //sorted in opposite directions
    //
    // This is the part that start at step f. The previous loop gave us a single
    // bitonic series, and now we're going to turn that into a sorted series
    //
    // I think ddd is the current direction. (blockIdx.x & 1) evaluates to 1
    // if the number is odd and 0 if the number is even, not that this stays
    // the same during half of the algorithm, and does not change at each
    // iteration
    //
    // note that this line is the only difference between this kernel and
    // the previous kernel, and it accounts for alternating directions of
    // sorting between pairs of contiguous blocks
    Direction ddd = (Direction)( blockIdx.x & 1 );

    // the stride is the distance (in indices) between the two keys being
    // compared, the comparator network for bitonic sort is B_n where n is
    // the stride length. Each thread t in the network compares elements at
    // t and t+stride

    // we iterate the stride starting at half of the size of the problem
    // and dividing by two at each step
    for(uint_t stride = sharedLength / 2; stride > 0; stride >>= 1)
    {
        // the comparator network is applied to the results of the previous
        // iteration so we must sync threads at each iteration
        __syncthreads();

        // this math is a little wonky bit it shuld resolve to make each
        // thread start at the correct place for it's comparator
        uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

        // perform the actual compare/swap
        compareSwap(
            s_key[pos +      0], s_val[pos +      0],
            s_key[pos + stride], s_val[pos + stride],
            ddd
        );
    }

    // perform one last sync before copying things out to global memory;
    __syncthreads();

    d_DstKey[                 0] = s_key[threadIdx.x +                  0];
    d_DstVal[                 0] = s_val[threadIdx.x +                  0];
    d_DstKey[(sharedLength / 2)] = s_key[threadIdx.x + (sharedLength / 2)];
    d_DstVal[(sharedLength / 2)] = s_val[threadIdx.x + (sharedLength / 2)];
}




/// bottom level of the bitonic sort
/**
 *  Since this kernel works in shared memory we'd like to use it as much as
 *  possible, so what we do is divide up the entire array that we want to sort
 *  and only sort it in sections that are small enough to fit in shared
 *  memory. However, we sort every other block in a different direction.
 *  As a result, each pair of results forms a bitonic series. We can then
 *  efficiently merge each pair of blocks into a sorted series, which we
 *  continue doing until the entire array is sorted.
 *
 *  Note: the next stage (Bitonic merge) accepts both ascending | descending
 *  and descending | ascending bitonic series
 *
 *  \ingroup    bitonic
 */
template <typename KeyType>
__global__ void sortSharedInc(
    KeyType     *d_DstKey,
    KeyType     *d_SrcKey,
    uint_t      sharedLength
){
    //Shared memory storage for the shared vectors, in the kernel call this
    //array is allocated to have
    //sharedLength*(sizeof(KeyType)+sizeof(ValueType)) bytes
    extern __shared__ unsigned int array[];

    // we have to generate pointers into the shared memory block for our actual
    // shared array storage
    KeyType*     s_key = (KeyType*)array;

    // calculate this threads offset to the data
    d_SrcKey += blockIdx.x * sharedLength + threadIdx.x;
    d_DstKey += blockIdx.x * sharedLength + threadIdx.x;

    // copy this threads data into shared memory
    s_key[threadIdx.x +                  0] = d_SrcKey[                 0];
    s_key[threadIdx.x + (sharedLength / 2)] = d_SrcKey[(sharedLength / 2)];


    // results in a tritonic series that can be split into two bitonic
    // serieses, i.e. steps a-e
    //
    // take a look at the comparator network drawing for the bitonic sort, this
    // first part is the first half of the network, we start by comparing only
    // pairs, (size=2), then we compaire quads (size=4)
    for(uint_t size = 2; size < sharedLength; size <<= 1)
    {
        // recall that size is always a power of two, so only one bit is
        // set to 1, so we shift that bit to the right by 1 and ask if that
        // new bit location is set in the thread id. If it is then the
        // direction is 1 otherwise it's 0
        Direction ddd = (Direction)( (threadIdx.x & (size / 2)) != 0 );

        // the stride is the distance (in indices) between the two keys being
        // compared.
        for(uint_t stride = size / 2; stride > 0; stride >>= 1)
        {
            // sync threads so that reads are not stale from the location of
            // another threads writes
            __syncthreads();

            // this math is really wonky, but I'm guessing that it resolves to
            // be the left side of the comparator network, i.e. when size = 2
            // then it does (x,y)->a, when size = 4 then it does a->(b,c) and
            // (b,c)->d, if the length of the example where bigger, this would
            // do more, but in the example this ends at e
            uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

            // perform the comparison/swap
            compareSwap(
                s_key[pos +      0],
                s_key[pos + stride],
                ddd
            );
        }
    }


    //Odd / even arrays of sharedLength elements
    //sorted in opposite directions
    //
    // This is the part that start at step f. The previous loop gave us a single
    // bitonic series, and now we're going to turn that into a sorted series
    //
    // I think ddd is the current direction. (blockIdx.x & 1) evaluates to 1
    // if the number is odd and 0 if the number is even, not that this stays
    // the same during half of the algorithm, and does not change at each
    // iteration
    //
    // note that this line is the only difference between this kernel and
    // the previous kernel, and it accounts for alternating directions of
    // sorting between pairs of contiguous blocks
    Direction ddd = (Direction)( blockIdx.x & 1 );

    // the stride is the distance (in indices) between the two keys being
    // compared, the comparator network for bitonic sort is B_n where n is
    // the stride length. Each thread t in the network compares elements at
    // t and t+stride

    // we iterate the stride starting at half of the size of the problem
    // and dividing by two at each step
    for(uint_t stride = sharedLength / 2; stride > 0; stride >>= 1)
    {
        // the comparator network is applied to the results of the previous
        // iteration so we must sync threads at each iteration
        __syncthreads();

        // this math is a little wonky bit it shuld resolve to make each
        // thread start at the correct place for it's comparator
        uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

        // perform the actual compare/swap
        compareSwap(
            s_key[pos +      0],
            s_key[pos + stride],
            ddd
        );
    }

    // perform one last sync before copying things out to global memory;
    __syncthreads();

    d_DstKey[                 0] = s_key[threadIdx.x +                  0];
    d_DstKey[(sharedLength / 2)] = s_key[threadIdx.x + (sharedLength / 2)];
}




/// sorts a bitonic series, this kernel is for a stride >= SHARED_SIZE_LIMIT
/**
 *  \ingroup    bitonic
 *
 *  \param[out] d_DstKey    array of sorted keys
 *  \param[out] d_DstVal    array of sorted values
 *  \param[in]  d_SrcKey    bitonic array with split at size/2
 *  \param[in]  d_SrcVal    values associated with d_SrcKey
 *  \param[in]  arrayLength the length of each array to sort
 *  \param[in]  dir         whether we should sort in ascending or descending
 *
 *  If A is an ascending sorted array and B is a descending sorted array, then
 *  [A,B] is a bitonic array. Here we merge A and B into a single sorted array
 *
 *  Note: usually this kernel is called with d_Dst... = d_Src... as it just
 *  merges results already stored in the destination buffer
 *
 *  Note: this kernel is used when the stride is too large to copy everything
 *  into shared memory, each thread just copies the two values they need to
 *  compare into global memory, performs the comparison/swap and then writes
 *  the results back to global memory
 *
 */
template <typename KeyType, typename ValueType>
__global__ void mergeGlobal(
    KeyType     *d_DstKey,
    ValueType   *d_DstVal,
    KeyType     *d_SrcKey,
    ValueType   *d_SrcVal,
    uint_t      arrayLength,
    uint_t      size,
    uint_t      stride,
    Direction   dir
){
    // the index of the comparator that this thread is acting as
    uint_t global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;

    // ?? the index of the comparator within it's own network ??
    uint_t        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

    // wtf? I guess this some how determines the direction that the comparator
    // should go for this thread
    Direction   ddd = (Direction)(
                            ((unsigned int)dir)
                            ^( (comparatorI & (size / 2)) != 0 )
                        );

    // calculate the position in the global array that this comparator needs
    // to start with
    uint_t pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    // copy out the two keys and values it needs to work with
    KeyType     keyA = d_SrcKey[pos +      0];
    ValueType   valA = d_SrcVal[pos +      0];
    KeyType     keyB = d_SrcKey[pos + stride];
    ValueType   valB = d_SrcVal[pos + stride];

    // perform the swap if necessary
    compareSwap(
        keyA, valA,
        keyB, valB,
        ddd
    );

    // write the (potentially swapped) results back to global memory
    d_DstKey[pos +      0] = keyA;
    d_DstVal[pos +      0] = valA;
    d_DstKey[pos + stride] = keyB;
    d_DstVal[pos + stride] = valB;
}




/// sorts a bitonic series, this kernel is for a stride >= SHARED_SIZE_LIMIT
/**
 *  \ingroup    bitonic
 *
 *  \param[out] d_DstKey    array of sorted keys
 *  \param[in]  d_SrcKey    bitonic array with split at size/2
 *  \param[in]  arrayLength the length of each array to sort
 *  \param[in]  dir         whether we should sort in ascending or descending
 *
 *  If A is an ascending sorted array and B is a descending sorted array, then
 *  [A,B] is a bitonic array. Here we merge A and B into a single sorted array
 *
 *  Note: usually this kernel is called with d_Dst... = d_Src... as it just
 *  merges results already stored in the destination buffer
 *
 *  Note: this kernel is used when the stride is too large to copy everything
 *  into shared memory, each thread just copies the two values they need to
 *  compare into global memory, performs the comparison/swap and then writes
 *  the results back to global memory
 *
 */
template <typename KeyType>
__global__ void mergeGlobal(
    KeyType     *d_DstKey,
    KeyType     *d_SrcKey,
    uint_t      arrayLength,
    uint_t      size,
    uint_t      stride,
    Direction   dir
){
    // the index of the comparator that this thread is acting as
    uint_t global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;

    // ?? the index of the comparator within it's own network ??
    uint_t        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

    // wtf? I guess this some how determines the direction that the comparator
    // should go for this thread
    Direction   ddd = (Direction)(
                            ((unsigned int)dir)
                            ^( (comparatorI & (size / 2)) != 0 )
                        );

    // calculate the position in the global array that this comparator needs
    // to start with
    uint_t pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    // copy out the two keys and values it needs to work with
    KeyType     keyA = d_SrcKey[pos +      0];
    KeyType     keyB = d_SrcKey[pos + stride];

    // perform the swap if necessary
    compareSwap(
        keyA,
        keyB,
        ddd
    );

    // write the (potentially swapped) results back to global memory
    d_DstKey[pos +      0] = keyA;
    d_DstKey[pos + stride] = keyB;
}




/// sorts a bitonic series, this kernel is for size > SHARED_SIZE_LIMIT and for
/// a stride in [1, SHARED_SIZE_LIMIT/2]
/**
 *  \ingroup    bitonic
 *
 *  \param[out] d_DstKey    array of sorted keys
 *  \param[out] d_DstVal    array of sorted values
 *  \param[in]  d_SrcKey    bitonic array with split at size/2
 *  \param[in]  d_SrcVal    values associated with d_SrcKey
 *  \param[in]  arrayLength the length of each array to sort
 *  \param[in]  size        the stride betwen two elements a comparator works on
 *  \param[in]  dir         whether we should sort in ascending or descending
 *
 *  If A is an ascending sorted array and B is a descending sorted array, then
 *  [A,B] is a bitonic array. Here we merge A and B into a single sorted array
 *
 *  Note: usually this kernel is called with d_Dst... = d_Src... as it just
 *  merges results already stored in the destination buffer
 *
 *  Note: this kernel is used when the stride is small enough to copy everything
 *  it needs into global memory. It is assumed that the stride is actually
 *  SHARED_SIZE_LIMIT/2 (exactly) and this kernel will perform all iterations
 *  for strides smaller than the initial until stride = 0;
 */
template <typename KeyType, typename ValueType>
__global__ void mergeShared(
    KeyType     *d_DstKey,
    ValueType   *d_DstVal,
    KeyType     *d_SrcKey,
    ValueType   *d_SrcVal,
    uint_t      arrayLength,
    uint_t      sharedLength,
    uint_t      size,
    Direction   dir
){
    //Shared memory storage for the shared vectors, in the kernel call this
    //array is allocated to have
    //sharedLength*(sizeof(KeyType)+sizeof(ValueType)) bytes
    extern __shared__ unsigned int array[];

    // we have to generate pointers into the shared memory block for our actual
    // shared array storage
    KeyType*     s_key = (KeyType*)array;
    ValueType*   s_val = (ValueType*)&s_key[sharedLength];

    // calculate the offset that this thread needs to start at
    d_SrcKey += blockIdx.x * sharedLength + threadIdx.x;
    d_SrcVal += blockIdx.x * sharedLength + threadIdx.x;
    d_DstKey += blockIdx.x * sharedLength + threadIdx.x;
    d_DstVal += blockIdx.x * sharedLength + threadIdx.x;

    // copy this threads value into shared (block) memory
    s_key[threadIdx.x +                  0] = d_SrcKey[                 0];
    s_val[threadIdx.x +                  0] = d_SrcVal[                 0];
    s_key[threadIdx.x + (sharedLength / 2)] = d_SrcKey[(sharedLength / 2)];
    s_val[threadIdx.x + (sharedLength / 2)] = d_SrcVal[(sharedLength / 2)];

    // calculate the index of this comparator
    uint_t comparatorI = UMAD(blockIdx.x, blockDim.x, threadIdx.x)
                            & ((arrayLength / 2) - 1);

    // determine the direction that this subarray needs to be sorted in
    Direction ddd = (Direction)(
                        ((unsigned int)dir)
                        ^( (comparatorI & (size / 2)) != 0 )
                    );

    // iterate over all remaining strides
    for(uint_t stride = sharedLength / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        // cacluate the position of this comparator
        uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

        // perform the swap
        compareSwap(
            s_key[pos +      0], s_val[pos +      0],
            s_key[pos + stride], s_val[pos + stride],
            ddd
        );
    }

    __syncthreads();

    // copy results back out to global memory
    d_DstKey[                 0] = s_key[threadIdx.x +                  0];
    d_DstVal[                 0] = s_val[threadIdx.x +                  0];
    d_DstKey[(sharedLength / 2)] = s_key[threadIdx.x + (sharedLength / 2)];
    d_DstVal[(sharedLength / 2)] = s_val[threadIdx.x + (sharedLength / 2)];
}




/// sorts a bitonic series, this kernel is for size > SHARED_SIZE_LIMIT and for
/// a stride in [1, SHARED_SIZE_LIMIT/2]
/**
 *  \ingroup    bitonic
 *
 *  \param[out] d_DstKey    array of sorted keys
 *  \param[in]  d_SrcKey    bitonic array with split at size/2
 *  \param[in]  arrayLength the length of each array to sort
 *  \param[in]  size        the stride betwen two elements a comparator works on
 *  \param[in]  dir         whether we should sort in ascending or descending
 *
 *  If A is an ascending sorted array and B is a descending sorted array, then
 *  [A,B] is a bitonic array. Here we merge A and B into a single sorted array
 *
 *  Note: usually this kernel is called with d_Dst... = d_Src... as it just
 *  merges results already stored in the destination buffer
 *
 *  Note: this kernel is used when the stride is small enough to copy everything
 *  it needs into global memory. It is assumed that the stride is actually
 *  SHARED_SIZE_LIMIT/2 (exactly) and this kernel will perform all iterations
 *  for strides smaller than the initial until stride = 0;
 */
template <typename KeyType>
__global__ void mergeShared(
    KeyType     *d_DstKey,
    KeyType     *d_SrcKey,
    uint_t      arrayLength,
    uint_t      sharedLength,
    uint_t      size,
    Direction   dir
){
    //Shared memory storage for the shared vectors, in the kernel call this
    //array is allocated to have
    //sharedLength*(sizeof(KeyType)+sizeof(ValueType)) bytes
    extern __shared__ unsigned int array[];

    // we have to generate pointers into the shared memory block for our actual
    // shared array storage
    KeyType*     s_key = (KeyType*)array;

    // calculate the offset that this thread needs to start at
    d_SrcKey += blockIdx.x * sharedLength + threadIdx.x;
    d_DstKey += blockIdx.x * sharedLength + threadIdx.x;

    // copy this threads value into shared (block) memory
    s_key[threadIdx.x +                  0] = d_SrcKey[                 0];
    s_key[threadIdx.x + (sharedLength / 2)] = d_SrcKey[(sharedLength / 2)];

    // calculate the index of this comparator
    uint_t comparatorI = UMAD(blockIdx.x, blockDim.x, threadIdx.x)
                            & ((arrayLength / 2) - 1);

    // determine the direction that this subarray needs to be sorted in
    Direction ddd = (Direction)(
                        ((unsigned int)dir)
                        ^( (comparatorI & (size / 2)) != 0 )
                    );

    // iterate over all remaining strides
    for(uint_t stride = sharedLength / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        // cacluate the position of this comparator
        uint_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

        // perform the swap
        compareSwap(
            s_key[pos +      0],
            s_key[pos + stride],
            ddd
        );
    }

    __syncthreads();

    // copy results back out to global memory
    d_DstKey[                 0] = s_key[threadIdx.x +                  0];
    d_DstKey[(sharedLength / 2)] = s_key[threadIdx.x + (sharedLength / 2)];
}




/// kernel launcher, sorts an array of key/value pairs using the bitonic sort
/// algorithm
/**
 *  \ingroup    bitonic
 *
 *  \param[out] d_DstKey        array of sorted keys
 *  \param[out] d_DstVal        array of sorted values
 *  \param[in]  d_SrcKey        array of unsorted keys
 *  \param[in]  d_SrcVal        array of unsorted values
 *  \param[in]  arrayLength     the length of each array to sort
 *  \param[in]  sharedLength    number of elements to store in shared arrays
 *  \param[in]  dir             whether we should sort ascending or descending
 *  \param[in]  globalThread    number of threads per block for global merge
 *
 *  \see http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm
 */
template <typename KeyType, typename ValueType>
uint_t sort(
    KeyType     *d_DstKey,      //< pointer to start of output array for keys
    ValueType   *d_DstVal,      //< pointer to start of output array for values
    KeyType     *d_SrcKey,      //< pointer to start of intput array for keys
    ValueType   *d_SrcVal,      //< pointer to start of input array for values
    uint_t      arrayLength,    //< the size of the array to sort
    uint_t      sharedLength,   //< number of shared memory elements per kernel
    Direction   dir,            //< whether to sort ascending or descending
    uint_t      globalThread    //< how many threads to use in global merge
){
    uint_t ret;

    //Nothing to sort
    if(arrayLength < 2)
        return 0;

    // fail if the the arrayLength is not a power of 2
    // note that, for future version, we just need to find the next power of
    // two and then pretend that all values greater than the array length are
    // infinite
    assert( isPow2(arrayLength) );
    assert( isPow2(sharedLength) );

    // if the array length is smaller than the shared size limit, then we only
    // need to do the sort kernel
    if(arrayLength <= sharedLength)
    {
        uint_t blockCount   = 1;
        uint_t threadCount  = arrayLength / 2;
        uint_t sharedMem    = arrayLength * ( sizeof(KeyType)
                                                + sizeof(ValueType) );

        // fail if the number of items is not a multiple of the shared size
        // limit... This appears to only be a simplification for the SDK
        // example, the algorithm generalizes to arrays which are not multiples
        // of the block size, we just need to fix that later
        sortShared<<<blockCount, threadCount, sharedMem>>>(
                d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);

        ret = threadCount;
    }

    // if the array lenght is larger than the shared size limit, then we need
    // to split the array up into blocks. We call the sort kernel on each
    // block which results in each block being sorted, note that they're
    // sorted in opposite directions though so that each pair forms a bitonic
    // series (i.e. increasing up to the break, then decreasing after)
    else
    {
        // note that right here we are assuming that each array is a multiple
        // of sharedLength, however, note that we enforce that arrayLength and
        // shared length are both powers of two, so if
        // arrayLength is >= sharedLength then blockCount will be a whole number
        // (in fact, a power of two)
        uint_t  blockCount  = dividePow2(arrayLength,sharedLength);

        // the number of threads we need is exactly 1/2 the size of the shared
        // buffer
        uint_t threadCount  = sharedLength / 2;

        // the amount of shared memory the kernel needs is given by the length
        // of the shared array it will sort times the amount of data in
        // each element
        uint_t sharedMem    = sharedLength * ( sizeof(KeyType)
                                                + sizeof(ValueType) );

        // this kernel sorts each block of the array into a bitonic list
        sortSharedInc<<<blockCount, threadCount,sharedMem>>>(
                        d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, sharedLength);

        // now we go through and iteratively merge the results
        // size starts at 2* the memory of a block and is double at each
        // iteration until it reaches the size of the array
        // note that the array composed of length (2*blockSize) composed of
        // concatinating two block results together forms a bitonic series
        //
        // note that this process is basically the same as the second half of
        // the sort kernel, however we have to do it by iteratively calling
        // the kernel because the resuls are all stored in global memory
        for(uint_t size = 2 * sharedLength; size <= arrayLength; size <<= 1)
        {
            // the stride starts at half of the size and is divided by two until
            // it reaches 0
            for(unsigned stride = size / 2; stride > 0; stride >>= 1)
            {
                // if the stride is too large, we used the merge kerenel that
                // works on global memory, this kernel only performs one
                // iteration of the merge (i.e. e->f and no further), this
                // is because the stride is so long that it crosses block
                // boundaries
                if(stride >= sharedLength)
                {
                    uint_t threadCount = std::min(globalThread,arrayLength/2);
                    uint_t blockCount  = arrayLength / (2*threadCount);
                    mergeGlobal<<<blockCount, threadCount>>>(
                            d_DstKey, d_DstVal, d_DstKey, d_DstVal,
                            arrayLength, size, stride, dir);
                }

                // if the stride is small enough, the comparator separation is
                // small enough that it will not cross thread boundaries (i.e.
                // see how f->(g,h) has no comparator crossing the half-way
                // mark)
                else
                {
                    mergeShared<<<blockCount, threadCount, sharedMem>>>(
                            d_DstKey, d_DstVal, d_DstKey, d_DstVal,
                            arrayLength, sharedLength, size, dir);

                    // this is important, note the break here, it should exit
                    // us as soon as we used the shared kernel, and consumes
                    // the rest of the for loop.. this kernel will loop through
                    // all remaining smaller strides
                    break;
                }
            }
        }

        ret = threadCount;
    }

    return ret;
}




/// kernel launcher, sorts an array of key/value pairs using the bitonic sort
/// algorithm
/**
 *  \ingroup    bitonic
 *
 *  \param[out] d_DstKey        array of sorted keys
 *  \param[in]  d_SrcKey        array of unsorted keys
 *  \param[in]  arrayLength     the length of each array to sort
 *  \param[in]  sharedLength    number of elements to store in shared arrays
 *  \param[in]  dir             whether we should sort ascending or descending
 *  \param[in]  globalThread    number of threads per block for global merge
 *
 *  \see http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm
 */
template <typename KeyType>
uint_t sort(
    KeyType     *d_DstKey,      //< pointer to start of output array for keys
    KeyType     *d_SrcKey,      //< pointer to start of intput array for keys
    uint_t      arrayLength,    //< the size of the array to sort
    uint_t      sharedLength,   //< number of shared memory elements per kernel
    Direction   dir,            //< whether to sort ascending or descending
    uint_t      globalThread    //< how many threads to use in global merge
){
    uint_t ret;

    //Nothing to sort
    if(arrayLength < 2)
        return 0;

    // fail if the the arrayLength is not a power of 2
    // note that, for future version, we just need to find the next power of
    // two and then pretend that all values greater than the array length are
    // infinite
    assert( isPow2(arrayLength) );
    assert( isPow2(sharedLength) );

    // if the array length is smaller than the shared size limit, then we only
    // need to do the sort kernel
    if(arrayLength <= sharedLength)
    {
        uint_t blockCount   = 1;
        uint_t threadCount  = arrayLength / 2;
        uint_t sharedMem    = arrayLength * ( sizeof(KeyType) );

        // fail if the number of items is not a multiple of the shared size
        // limit... This appears to only be a simplification for the SDK
        // example, the algorithm generalizes to arrays which are not multiples
        // of the block size, we just need to fix that later
        sortShared<<<blockCount, threadCount, sharedMem>>>(
                d_DstKey, d_SrcKey, arrayLength, dir);

        ret = threadCount;
    }

    // if the array lenght is larger than the shared size limit, then we need
    // to split the array up into blocks. We call the sort kernel on each
    // block which results in each block being sorted, note that they're
    // sorted in opposite directions though so that each pair forms a bitonic
    // series (i.e. increasing up to the break, then decreasing after)
    else
    {
        // note that right here we are assuming that each array is a multiple
        // of sharedLength, however, note that we enforce that arrayLength and
        // shared length are both powers of two, so if
        // arrayLength is >= sharedLength then blockCount will be a whole number
        // (in fact, a power of two)
        uint_t  blockCount  = dividePow2(arrayLength,sharedLength);

        // the number of threads we need is exactly 1/2 the size of the shared
        // buffer
        uint_t threadCount  = sharedLength / 2;

        // the amount of shared memory the kernel needs is given by the length
        // of the shared array it will sort times the amount of data in
        // each element
        uint_t sharedMem    = sharedLength * ( sizeof(KeyType) );

        // this kernel sorts each block of the array into a bitonic list
        sortSharedInc<<<blockCount, threadCount,sharedMem>>>(
                        d_DstKey, d_SrcKey, sharedLength);
        cuda::checkLastError("bitonic::sortSharedInc");

        // now we go through and iteratively merge the results
        // size starts at 2* the memory of a block and is double at each
        // iteration until it reaches the size of the array
        // note that the array composed of length (2*blockSize) composed of
        // concatinating two block results together forms a bitonic series
        //
        // note that this process is basically the same as the second half of
        // the sort kernel, however we have to do it by iteratively calling
        // the kernel because the resuls are all stored in global memory
        for(uint_t size = 2 * sharedLength; size <= arrayLength; size <<= 1)
        {
            // the stride starts at half of the size and is divided by two until
            // it reaches 0
            for(unsigned stride = size / 2; stride > 0; stride >>= 1)
            {
                // if the stride is too large, we used the merge kerenel that
                // works on global memory, this kernel only performs one
                // iteration of the merge (i.e. e->f and no further), this
                // is because the stride is so long that it crosses block
                // boundaries
                if(stride >= sharedLength)
                {
                    uint_t threadCount = std::min(globalThread,arrayLength/2);
                    uint_t blockCount  = arrayLength / (2*threadCount);
                    mergeGlobal<<<blockCount, threadCount>>>(
                            d_DstKey,d_DstKey,
                            arrayLength, size, stride, dir);
                    cuda::checkLastError("bitonic::mergeGlobal");
                }

                // if the stride is small enough, the comparator separation is
                // small enough that it will not cross thread boundaries (i.e.
                // see how f->(g,h) has no comparator crossing the half-way
                // mark)
                else
                {
                    mergeShared<<<blockCount, threadCount, sharedMem>>>(
                            d_DstKey, d_DstKey,
                            arrayLength, sharedLength, size, dir);
                    cuda::checkLastError("bitonic::mergeShared");

                    // this is important, note the break here, it should exit
                    // us as soon as we used the shared kernel, and consumes
                    // the rest of the for loop.. this kernel will loop through
                    // all remaining smaller strides
                    break;
                }
            }
        }

        ret = threadCount;
    }

    return ret;
}




/// used when arrayLength is not a power of two, it writes \init to all values
/// of d_SrcKey (which is an offset from of the actual source buffer)
/**
 *  \ingroup    bitonic
 *  \param[in]  d_SrcKey        offset with buffer where padding starts
 *  \param[in]  init            value to write to all the overflow keys
 *  \param[in]  arrayLength     number of values to write
 */
template <typename KeyType>
__global__ void prepare( KeyType*    d_SrcKey,
                         KeyType     init,
                         uint_t      arrayLength
                        )
{
    uint_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < arrayLength)
        d_SrcKey[tid] = init;
}










} // namespace bitonic
} // namespace cuda
} // namespace mpblocks



#endif
