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
 *  @file   dubins/curves_cuda/PointSet.hpp
 *
 *  @date   Jun 18, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDANN_POINTSET_HPP_
#define MPBLOCKS_CUDANN_POINTSET_HPP_


#include <map>
#include <string>
#include <limits>
#include <iostream>

#include <mpblocks/cuda.hpp>
#include <mpblocks/cudaNN/PointSet.h>

namespace mpblocks {
namespace   cudaNN {




template <typename Format_t, unsigned int NDim>
void ResultBlock<Format_t,NDim>::allocate( uint_t cols )
{
    if(m_buf)
        delete [] m_buf;

    m_buf = new Format_t[NDim*cols];
    m_cols  = cols;
}










template <typename Format_t, unsigned int NDim>
PointSet<Format_t,NDim>::PointSet(uint_t n):
    m_sorter( -std::numeric_limits<Format_t>::max(),
               std::numeric_limits<Format_t>::max() )
{
    m_g_in    = 0;
    m_g_out   = 0;
    m_g_sorted= 0;

    m_threadsPerBlock = 0;
    m_nSM             = 0;

    deallocate();

    try
    {
        config();
        allocate(n);
    }
    catch( const std::exception& ex )
    {
        std::cerr << "Error in constructing dubins CUDA PointSet: "
                  << ex.what()
                  << "\nNote: point set is unallocated\n";
    }
}










template <typename Format_t, unsigned int NDim>
PointSet<Format_t,NDim>::~PointSet()
{
    deallocate();
}










template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::deallocate()
{
    if(m_g_in)
    {
        cuda::free(m_g_in);
        m_g_in = 0;
    }

    if(m_g_out)
    {
        cuda::free(m_g_out);
        m_g_out = 0;
    }

    if(m_g_sorted)
    {
        cuda::free(m_g_sorted);
        m_g_sorted = 0;
    }

    m_dbAlloc  = 0;
    m_dbAlloc2 = 0;
    m_dbSize   = 0;
}










template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::allocate(uint_t n)
{
    deallocate();

    m_dbAlloc   = n;
    m_dbAlloc2  = cuda::nextPow2(n);

    m_g_in  = cuda::mallocPitchT<Format_t>( m_pitchIn, m_dbAlloc, NDim );
    std::cout << "allocated m_g_in for "
                  << m_dbAlloc << " object with pitch: " << m_pitchIn << "\n";

    m_g_out = cuda::mallocPitchT<Format_t>( m_pitchOut, m_dbAlloc2, 2 );
    std::cout << "allocated m_g_out for "
                  << m_dbAlloc << " object with pitch: " << m_pitchOut << "\n";

    m_g_sorted = cuda::mallocPitchT<Format_t>( m_pitchOut, m_dbAlloc2, 2 );
    std::cout << "allocated m_g_sorted for "
                  << m_dbAlloc << " object with pitch: " << m_pitchOut << "\n";
}


















template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::clear(bool clearmem)
{
    m_dbSize = 0;
    if( clearmem )
    {
        cuda::memset2DT( m_g_in,     m_pitchIn,  0, m_dbAlloc,  NDim );
        cuda::memset2DT( m_g_out,    m_pitchOut, 0, m_dbAlloc2, 2 );
        cuda::memset2DT( m_g_sorted, m_pitchOut, 0, m_dbAlloc2, 2 );
    }
}










template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::config()
{
    int devId = cuda::getDevice();
    config(devId);
}









template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::computeGrid( uint_t& blocks, uint_t& threads )
{
    threads = cuda::intDivideRoundUp(m_dbSize,m_nSM);
    if( threads > m_threadsPerBlock )
        threads = m_threadsPerBlock;
    blocks  = cuda::intDivideRoundUp(m_dbSize,threads);
}










template <typename Format_t, unsigned int NDim>
int PointSet<Format_t,NDim>::insert( const Format_t q[NDim] )
{
    cuda::memcpy2DT( m_g_in + m_dbSize, m_pitchIn,
                     q, sizeof(Format_t),
                     1, NDim,
                     cudaMemcpyHostToDevice );
    m_dbSize++;
    return m_dbSize-1;
}









} //< namespace cudaNN
} //< namespace mpblocks


#endif // POINTSET_HPP_
