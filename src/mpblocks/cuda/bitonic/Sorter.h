/*
 *  Copyright (C) 2013 Josh Bialkowski (jbialk@mit.edu)
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
 *  @file   Sorter.h
 *
 *  @date   Sep 3, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  class definition for bitonic sort class
 */

#ifndef MPBLOCKS_CUDA_BITONIC_SORTER_H_
#define MPBLOCKS_CUDA_BITONIC_SORTER_H_

/**
 *  @defgroup bitonic Bitonic Sorting
 *
 *  Contains all the GPU (CUDA) kernels used to implement the bitonic sort
 *  algorithm, as well as a driver class for launching sort operations
 *
 *  @see    http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm
 */


namespace mpblocks {
namespace     cuda {
namespace  bitonic {


/// A utility class for calculating properties of the bitonic sort kernels
/**
 *  The sorter object calculates how much shared memory and how many threads
 *  the kernels should use to maximize efficiency. These values are calculated
 *  based on attributes of the kerenels and compared to the specifics of the
 *  active device at runtime. These values are used to size the grid and the
 *  blocks in the grid for the kernel calls
 *
 *  Note that the bitonic sort algorithm only works for arrays which are
 *  power-of-two in length. The Sorter class will automatically grow smaller
 *  arrays into the next power-of-two so that the bitonic sort algorithm can
 *  be used
 *
 *  Note that \p KeyType must have a specialization of std::numeric_limits or
 *  the prepare kernel wont know how to fill in the rest of the array. There
 *  must also exist an operator< (less-than operator, i.e. "<") defined for
 *  \p KeyType
 *
 *  \tparam     KeyType     the type (class) that the keys to sort
 *  \tparam     ValueType   the type (class) of the values associated with keys
 *
 *  \see    http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm
 *
 *  \ingroup    bitonic
 */
template <typename KeyType, typename ValueType = void>
class Sorter
{
    public:
        typedef unsigned int uint_t;

    private:
        uint_t  m_sharedLength;     ///< the size (in elements) of shared memory
                                    ///  buffers in all the kernels (also, 2x
                                    ///  the number of threads in a block)
        uint_t  m_threadsPrepare;   ///< number of threads per block to use in
                                    ///  the prepare kernel
        uint_t  m_threadsMerge;     ///< number of threads per block to use in
                                    ///  the global merge kernel
        uint_t  m_nSM;              ///< number of SMs of the current device

        KeyType m_min;
        KeyType m_max;

        /// used when arrayLength is not a power-of-two, fills the remaining
        /// elements with either the min key value of the max key value
        void prepare(   KeyType*    d_SrcKey,
                        uint_t      arrayLength,
                        Direction   dir);


    public:
        /// the constructor queries the current device and calculates relevent
        /// parameters to decide how to size the kernels when they're called
        Sorter(KeyType min, KeyType max);

        /// configures the sorter for the current cuda device
        void config();

        /// configures the sorter for the specified cuda device
        void config( int dev );

        /// actually perform the sort
        /**
         *  @param[out] *d_DstKey       the output array for sorted keys
         *  @param[out] *d_DstVal       the output array for sorted values
         *  @param[in]  *d_SrcKey       pointer to start of intput array for keys
         *  @param[in]  *d_SrcVal       pointer to start of input array for values
         *  @param[in]  arrayLength     the size of the array to sort
         *  @param[in]  dir             whether to sort ascending or descending)
         */
        uint_t sort(
                KeyType     *d_DstKey,      //< the output array for sorted keys
                ValueType   *d_DstVal,      //< the output array for sorted values
                KeyType     *d_SrcKey,      //< pointer to start of intput array for keys
                ValueType   *d_SrcVal,      //< pointer to start of input array for values
                uint_t      arrayLength,    //< the size of the array to sort
                Direction   dir = Ascending //< whether to sort ascending or descending)
                );
};






/// partial specialization for key-only sort
template <typename KeyType>
class Sorter<KeyType,void>
{
    public:
        typedef unsigned int uint_t;

    private:
        uint_t  m_sharedLength;     ///< the size (in elements) of shared memory
                                    ///  buffers in all the kernels (also, 2x
                                    ///  the number of threads in a block)
        uint_t  m_threadsPrepare;   ///< number of threads per block to use in
                                    ///  the prepare kernel
        uint_t  m_threadsMerge;     ///< number of threads per block to use in
                                    ///  the global merge kernel
        uint_t  m_nSM;              ///< number of SMs of the current device

        KeyType m_min;
        KeyType m_max;

        /// used when arrayLength is not a power-of-two, fills the remaining
        /// elements with either the min key value of the max key value
        void prepare(   KeyType*    d_SrcKey,
                        uint_t      arrayLength,
                        Direction   dir );

    public:
        /// the constructor queries the current device and calculates relevent
        /// parameters to decide how to size the kernels when they're called
        Sorter(KeyType min, KeyType max);

        /// configures the sorter for the current cuda device
        void config();

        /// configures the sorter for the specified cuda device
        void config( int dev );

        /// actually perform the sort
        /**
         *  @param[out] *d_DstKey       the output array for sorted keys
         *  @param[out] *d_DstVal       the output array for sorted values
         *  @param[in]  *d_SrcKey       pointer to start of intput array for keys
         *  @param[in]  *d_SrcVal       pointer to start of input array for values
         *  @param[in]  arrayLength     the size of the array to sort
         *  @param[in]  dir             whether to sort ascending or descending)
         */
        uint_t sort(
                KeyType     *d_DstKey,      //< the output array for sorted keys
                KeyType     *d_SrcKey,      //< pointer to start of intput array for keys
                uint_t      arrayLength,    //< the size of the array to sort
                Direction   dir = Ascending //< whether to sort ascending or descending)
                );
};





} // namespace bitonic
} // namespace cuda
} // namespace mpblocks


#endif// BITONIC_SORTER_H
