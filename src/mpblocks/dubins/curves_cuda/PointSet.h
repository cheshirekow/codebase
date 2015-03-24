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

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_POINTSET_H_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_POINTSET_H_


#include <map>
#include <string>

namespace    mpblocks {
namespace      dubins {
namespace curves_cuda {


template <typename Format_t>
class ResultBlock
{
    public:
        typedef unsigned int uint_t;

    private:
        uint_t m_rows;
        uint_t m_cols;

        Format_t* m_buf;

    public:
        ResultBlock( uint_t rows=0, uint_t cols=0 ):
            m_rows(0),
            m_cols(0),
            m_buf(0)
        {
            allocate( rows, cols );
        }

        ~ResultBlock()
        {
            if( m_buf )
                delete [] m_buf;
        }

        void allocate( uint_t rows, uint_t cols );

        Format_t* ptr() const { return m_buf; }
        uint_t rows()  const { return m_rows; }
        uint_t cols()  const { return m_cols; }
        uint_t pitch() const { return m_cols*sizeof(Format_t); }

        Format_t operator()( int i, int j )
        {
            return m_buf[ i*m_cols + j ];
        }
};



/// provides a convenience interface for managing a point set of dubins states
/// in GPU memory, including brute force CUDA searches on that point set
template <typename Format_t>
class PointSet
{
    public:
        typedef unsigned int uint_t;
        typedef std::map< std::string, cuda::FuncAttributes >  fattrMap_t;
        typedef cuda::bitonic::Sorter<Format_t,Format_t> Sorter_t;

    private:
        Params<Format_t> m_params;   ///< query parameters
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
        PointSet(uint_t n=10, Format_t r=1);
        ~PointSet();

        /// deallocate and zero out pointers
        void deallocate();

        /// reallocates device storage for a point set of size n, also resets
        /// the database
        void allocate(uint_t n);

        /// set the radius
        void set_r( Format_t r );

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
        int  insert( Format_t q[3] );

        /// batch compute distance to point set
        void distance_to_set(  Format_t q[3], ResultBlock<Format_t>& out );

        /// batch compute distance from point set
        void distance_from_set(  Format_t q[3], ResultBlock<Format_t>& out  );

        /// return k nearest children of q
        void nearest_children( Format_t q[3], ResultBlock<Format_t>& out  );

        /// return k nearest parents of q
        void nearest_parents(  Format_t q[3], ResultBlock<Format_t>& out );

        /// batch compute euclidean distances
        void group_distance_to_set( Format_t q[3], ResultBlock<Format_t>& out );

        /// find k euclidean nearest neighbors
        void group_distance_neighbors( Format_t q[3], ResultBlock<Format_t>& out );

        /// retrieve kernel attributes into the map, intended only for printing
        /// out statistics
        static void get_fattr( fattrMap_t&  );
};



} // curves
} // dubins
} // mpblocks

#endif // POINTSET_H_
