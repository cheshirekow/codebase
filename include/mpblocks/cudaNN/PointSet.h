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
 *  @file   dubins/curves_cuda/PointSet.h
 *
 *  @date   Jun 13, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDANN_POINTSET_H_
#define MPBLOCKS_CUDANN_POINTSET_H_


#include <map>
#include <string>
#include <mpblocks/cuda/bitonic.h>

namespace mpblocks {
namespace   cudaNN {


template <typename Format_t, unsigned int NDim>
class ResultBlock
{
    public:
        typedef unsigned int uint_t;

    private:
        uint_t    m_cols;
        Format_t* m_buf;

    public:
        ResultBlock( uint_t cols=0 ):
            m_cols(0),
            m_buf(0)
        {
            allocate( cols );
        }

        ~ResultBlock()
        {
            if( m_buf )
                delete [] m_buf;
        }

        void allocate( uint_t cols );

        Format_t* ptr() const { return m_buf; }
        uint_t cols()  const { return m_cols; }
        uint_t pitch() const { return m_cols*sizeof(Format_t); }

        Format_t operator()( int i, int j )
        {
            return m_buf[ i*m_cols + j ];
        }
};


struct EuclideanTag{};
struct SE3Tag
{
    float w;
    SE3Tag( float w=1 ):w(w){}
    SE3Tag operator()( float w ) const{ return SE3Tag(w); }
};

struct R2S1Tag
{
    float w;
    R2S1Tag( float w=1):w(w){}
    R2S1Tag operator()( float w ) const{ return R2S1Tag(w); }
};

const EuclideanTag EUCLIDEAN;
const SE3Tag       SE3;
const R2S1Tag      R2S1;


/// provides a convenience interface for managing a point set
/// in GPU memory, and dispatching brute force CUDA searches on that point set
template <typename Format_t, unsigned int NDim>
class PointSet
{
    public:
        typedef unsigned int uint_t;
        typedef cuda::bitonic::Sorter<Format_t,Format_t> Sorter_t;
        typedef ResultBlock<Format_t,2> Result_t;

    private:
        uint_t    m_dbAlloc;    ///< size of the point set allocated
        uint_t    m_dbAlloc2;   ///< size allocated for the sorted set, will be
                                ///  the next power of two of dbAlloc
        uint_t    m_dbSize;     ///< size of the point set filled
        size_t    m_pitchIn;    ///< row-pitch of buffers (in *bytes*)
        size_t    m_pitchOut;   ///< row-pitch of buffers (in *bytes*)
        Format_t* m_g_in;       ///< kernel input buffer
        Format_t* m_g_out;      ///< kernel output buffer
        Format_t* m_g_sorted;   ///< output for sorted results

        Sorter_t  m_sorter;     ///< wraps sort kernels

        uint_t  m_threadsPerBlock;  ///< maximum threads per block
        uint_t  m_nSM;              ///< number of multiprocessors

    public:
        PointSet(uint_t n=10);
        ~PointSet();

        /// deallocate and zero out pointers
        void deallocate();

        /// reallocates device storage for a point set of size n, also resets
        /// the database
        void allocate(uint_t n);

        /// clear the database and reset input iterator
        void clear(bool clearmem=false);

        /// retreives device properties of the current device, used to calculate
        /// kernel peramaters, call once after setting the cuda device and
        /// before launching any kernels
        void config();

        /// retreives device properties of the specified device, used to calculate
        /// kernel peramaters, call once after setting the cuda device and
        /// before launching any kernels
        void config(int dev);

    private:
        /// compute the grid size given the current configuration and size of
        /// the point set
        void computeGrid( uint_t& blocks, uint_t& threads );

    public:
        /// insert a new state into the point set, and return it's id
        int  insert( const Format_t q[NDim] );

        /// batch compute distance to point set
        void distance( EuclideanTag, const Format_t q[NDim], Result_t& out );
        void distance( SE3Tag,       const Format_t q[NDim], Result_t& out );
        void distance( R2S1Tag,      const Format_t q[NDim], Result_t& out );

        void distance( EuclideanTag, const Format_t q[NDim], uint_t size, Result_t& out );
        void distance( SE3Tag,       const Format_t q[NDim], uint_t size, Result_t& out );
        void distance( R2S1Tag,      const Format_t q[NDim], uint_t size, Result_t& out );

        /// return k nearest children of q, k is columns of out
        void nearest( EuclideanTag, const Format_t q[NDim], Result_t& out );
        void nearest( SE3Tag,       const Format_t q[NDim], Result_t& out );
        void nearest( R2S1Tag,      const Format_t q[NDim], Result_t& out );

        void nearest( EuclideanTag, const Format_t q[NDim], uint_t size, Result_t& out );
        void nearest( SE3Tag,       const Format_t q[NDim], uint_t size, Result_t& out );
        void nearest( R2S1Tag,      const Format_t q[NDim], uint_t size, Result_t& out );
};



} //< namespace cudaNN
} //< namespace mpblocks

#endif // POINTSET_H_
