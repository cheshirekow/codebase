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

#ifndef MPBLOCKS_CUDANN_POINTSET_CU_HPP_
#define MPBLOCKS_CUDANN_POINTSET_CU_HPP_


#include <map>
#include <string>
#include <limits>
#include <iostream>

#include <mpblocks/cuda.hpp>
#include <mpblocks/cuda/bitonic.cu.hpp>
#include <mpblocks/cudaNN/PointSet.h>
#include <mpblocks/cudaNN/kernels.cu.hpp>


namespace mpblocks {
namespace   cudaNN {


template <typename Format_t, unsigned int NDim, bool Enable>
struct SE3Attr
{
    static void maxRegs(unsigned int &maxRegs){}
};

template <typename Format_t, unsigned int NDim>
struct SE3Attr<Format_t,NDim,true>
{
    static void maxRegs(unsigned int &maxRegs)
    {
        typedef unsigned int uint_t;
        cuda::FuncAttributes attr;

        attr.getFrom( &kernels::se3_distance<Format_t,NDim> );
        maxRegs = std::max(maxRegs, (uint_t)attr.numRegs);
    }
};



template <typename Format_t, unsigned int NDim, bool Enable>
struct R2S1Attr
{
    static void maxRegs(unsigned int &maxRegs){}
};

template <typename Format_t, unsigned int NDim>
struct R2S1Attr<Format_t,NDim,true>
{
    static void maxRegs(unsigned int &maxRegs)
    {
        typedef unsigned int uint_t;
        cuda::FuncAttributes attr;

        attr.getFrom( &kernels::r2s1_distance<Format_t,NDim> );
        maxRegs = std::max(maxRegs, (uint_t)attr.numRegs);
    }
};


template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::config(int devId)
{
    cuda::DeviceProp     devProps(devId);
    cuda::FuncAttributes attr;
    uint_t  maxRegs     = 0;

    typedef QueryPoint<Format_t,NDim>     QP;
    typedef RectangleQuery<Format_t,NDim> QR;
    typedef unsigned int                  uint;
    typedef void (*point_dist_fn)(QP,Format_t*,uint,Format_t*,uint,uint);
    point_dist_fn euclidean_dist_fn =
            &kernels::euclidean_distance<Format_t,NDim>;

    attr.getFrom( euclidean_dist_fn );
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);

    SE3Attr<Format_t, NDim,(NDim>=7)>::maxRegs(maxRegs);
    R2S1Attr<Format_t,NDim,(NDim>=3)>::maxRegs(maxRegs);

    // the maximum number of threads we can put into a block is given by the
    // number of registers on each SM divided by the number of registers that
    // are used by each thread in the kernel
    uint_t  threadCount_max     = (uint_t)devProps.regsPerBlock / maxRegs;

    // make sure that the number of threads per block computed as above doesn't
    // exceed the max per-block for the architectture
    m_threadsPerBlock = std::min( threadCount_max,
                                (uint_t)devProps.maxThreadsPerBlock);

    // get the number of multiprocessors
    m_nSM = devProps.multiProcessorCount;

    // configure the sorter
    m_sorter.config(devId);
}


template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::distance(
        EuclideanTag tag, const Format_t q[NDim], Result_t& out )
{
    distance(tag,q,m_dbSize,out);
}


template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::distance(
        EuclideanTag, const Format_t q[NDim], uint_t size, Result_t& out )
{
    uint_t blocks,threads;
    computeGrid(blocks,threads);

    size_t pitchIn  = m_pitchIn/sizeof(Format_t);
    size_t pitchOut = m_pitchOut/sizeof(Format_t);
    QueryPoint<Format_t,NDim> query;
    std::copy(q,q+NDim,query.data);

    // call the kernel
    kernels::euclidean_distance<Format_t,NDim><<<blocks,threads>>>(
        query,
        m_g_in,
        pitchIn,
        m_g_out,
        pitchOut,
        size
        );
    cuda::deviceSynchronize();

    // retrieve results
    cuda::memcpy2DT(
            out.ptr(),  out.pitch(),
            m_g_out,    m_pitchOut,
            out.cols(), 1,
            cudaMemcpyDeviceToHost );
}


template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::nearest(
        EuclideanTag tag, const Format_t q[NDim], Result_t& out  )
{
    nearest(tag,q,m_dbSize,out);
}


template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::nearest(
        EuclideanTag, const Format_t q[NDim], uint_t size, Result_t& out  )
{
    uint_t blocks,threads;
    computeGrid(blocks,threads);

    size_t pitchIn  = m_pitchIn/sizeof(Format_t);
    size_t pitchOut = m_pitchOut/sizeof(Format_t);
    QueryPoint<Format_t,NDim> query;
    std::copy(q,q+NDim,query.data);

    // call the kernel to calculate distances to children
    kernels::euclidean_distance<Format_t,NDim><<<blocks,threads>>>(
        query,
        m_g_in,
        pitchIn,
        m_g_out,
        pitchOut,
        size
        );
    cuda::deviceSynchronize();

    Format_t* unsortedKeys = m_g_out;
    Format_t* unsortedVals = m_g_out    + pitchOut;
    Format_t* sortedKeys   = m_g_sorted;
    Format_t* sortedVals   = m_g_sorted + pitchOut;

    // call the kernel to sort the results
    m_sorter.sort(
        sortedKeys,   sortedVals,
        unsortedKeys, unsortedVals,
        size,
        cuda::bitonic::Ascending );
    cuda::deviceSynchronize();

    // fetch the k smallest
    cuda::memcpy2DT(
        out.ptr(),  out.pitch(),
        m_g_sorted, m_pitchOut,
        out.cols(), 2,
        cudaMemcpyDeviceToHost );
}


template <typename Format_t, unsigned int NDim,bool Enabled>
struct SE3Kernel
{
    static void dispatch( Format_t     weight,
       const QueryPoint<Format_t,NDim>& q,
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n,
       unsigned int blocks,
       unsigned int threads )
    {
        std::cerr << "CANNOT CALL SE3 KERNEL IF NDIM != 7\n";
        assert( false );
    }
};




template <typename Format_t, unsigned int NDim>
struct SE3Kernel<Format_t,NDim,true>
{
    static void dispatch(
        Format_t     weight,
       const QueryPoint<Format_t,NDim>& q,
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n,
       unsigned int blocks,
       unsigned int threads)
    {
        // call the kernel
        kernels::se3_distance<Format_t,NDim><<<blocks,threads>>>
            (weight,q,g_in,pitchIn,g_out,pitchOut,n );
        cuda::deviceSynchronize();
    }
};


template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::distance(
        SE3Tag params, const Format_t q[NDim], Result_t& out )
{
    distance(params,q,m_dbSize,out);
}


template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::distance(
        SE3Tag params, const Format_t q[NDim], uint_t size, Result_t& out )
{
    uint_t blocks,threads;
    computeGrid(blocks,threads);

    size_t pitchIn  = m_pitchIn/sizeof(Format_t);
    size_t pitchOut = m_pitchOut/sizeof(Format_t);
    QueryPoint<Format_t,NDim> query;
    std::copy(q,q+NDim,query.data);
    Format_t w = params.w;

    SE3Kernel<Format_t,NDim,(NDim>=7)>::dispatch(
            w,
            query,
            m_g_in,
            pitchIn,
            m_g_out,
            pitchOut,
            size,
            blocks, threads);


    // retrieve results
    cuda::memcpy2DT(
            out.ptr(),  out.pitch(),
            m_g_out,    m_pitchOut,
            out.cols(), 1,
            cudaMemcpyDeviceToHost );
}


template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::nearest(
        SE3Tag params, const Format_t q[NDim], Result_t& out  )
{
    nearest(params,q,m_dbSize,out);
}


template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::nearest(
        SE3Tag params, const Format_t q[NDim], uint_t size, Result_t& out  )
{
    uint_t blocks,threads;
    computeGrid(blocks,threads);


    size_t pitchIn  = m_pitchIn/sizeof(Format_t);
    size_t pitchOut = m_pitchOut/sizeof(Format_t);
    QueryPoint<Format_t,NDim> query;
    std::copy(q,q+NDim,query.data);
    Format_t w = params.w;

    // call the kernel to calculate distances to children
    SE3Kernel<Format_t,NDim,(NDim>=7)>::dispatch(
            w,
            query,
            m_g_in,
            pitchIn,
            m_g_out,
            pitchOut,
            size,
            blocks, threads );

    Format_t* unsortedKeys = m_g_out;
    Format_t* unsortedVals = m_g_out    + pitchOut;
    Format_t* sortedKeys   = m_g_sorted;
    Format_t* sortedVals   = m_g_sorted + pitchOut;

    // call the kernel to sort the results
    m_sorter.sort(
        sortedKeys,   sortedVals,
        unsortedKeys, unsortedVals,
        size,
        cuda::bitonic::Ascending );
    cuda::deviceSynchronize();

    // fetch the k smallest
    cuda::memcpy2DT(
        out.ptr(),  out.pitch(),
        m_g_sorted, m_pitchOut,
        out.cols(), 2,
        cudaMemcpyDeviceToHost );

    cuda::deviceSynchronize();
}






template <typename Format_t, unsigned int NDim,bool Enabled>
struct R2S1Kernel
{
    static void dispatch( Format_t     weight,
       const QueryPoint<Format_t,NDim>& q,
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n,
       unsigned int blocks,
       unsigned int threads )
    {
        std::cerr << "CANNOT CALL R2S1 KERNEL IF NDIM < 3 (" << NDim << ")\n";
        assert( false );
    }
};




template <typename Format_t, unsigned int NDim>
struct R2S1Kernel<Format_t,NDim,true>
{
    static void dispatch(
       Format_t     weight,
       const QueryPoint<Format_t,NDim>& q,
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n,
       unsigned int blocks,
       unsigned int threads)
    {
        // call the kernel
        kernels::r2s1_distance<Format_t,NDim><<<blocks,threads>>>
            (weight,q,g_in,pitchIn,g_out,pitchOut,n );
        cuda::deviceSynchronize();
    }
};


template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::distance(
        R2S1Tag params, const Format_t q[NDim], Result_t& out )
{
    distance(params,q,m_dbSize,out);
}



template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::distance(
        R2S1Tag params, const Format_t q[NDim], uint_t size, Result_t& out )
{
    uint_t blocks,threads;
    computeGrid(blocks,threads);

    size_t pitchIn  = m_pitchIn/sizeof(Format_t);
    size_t pitchOut = m_pitchOut/sizeof(Format_t);
    QueryPoint<Format_t,NDim> query;
    std::copy(q,q+NDim,query.data);
    Format_t w = params.w;

    R2S1Kernel<Format_t,NDim,(NDim>=3)>::dispatch(
            w,
            query,
            m_g_in,
            pitchIn,
            m_g_out,
            pitchOut,
            size,
            blocks, threads);


    // retrieve results
    cuda::memcpy2DT(
            out.ptr(),  out.pitch(),
            m_g_out,    m_pitchOut,
            out.cols(), 1,
            cudaMemcpyDeviceToHost );
}


template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::nearest(
        R2S1Tag params, const Format_t q[NDim], Result_t& out  )
{
    nearest(params,q,m_dbSize,out);
}

template <typename Format_t, unsigned int NDim>
void PointSet<Format_t,NDim>::nearest(
        R2S1Tag params, const Format_t q[NDim], uint_t size, Result_t& out  )
{
    uint_t blocks,threads;
    computeGrid(blocks,threads);


    size_t pitchIn  = m_pitchIn/sizeof(Format_t);
    size_t pitchOut = m_pitchOut/sizeof(Format_t);
    QueryPoint<Format_t,NDim> query;
    std::copy(q,q+NDim,query.data);
    Format_t w = params.w;

    // call the kernel to calculate distances to children
    R2S1Kernel<Format_t,NDim,(NDim>=3)>::dispatch(
            w,
            query,
            m_g_in,
            pitchIn,
            m_g_out,
            pitchOut,
            size,
            blocks, threads );

    Format_t* unsortedKeys = m_g_out;
    Format_t* unsortedVals = m_g_out    + pitchOut;
    Format_t* sortedKeys   = m_g_sorted;
    Format_t* sortedVals   = m_g_sorted + pitchOut;

    // call the kernel to sort the results
    m_sorter.sort(
        sortedKeys,   sortedVals,
        unsortedKeys, unsortedVals,
        size,
        cuda::bitonic::Ascending );
    cuda::deviceSynchronize();

    // fetch the k smallest
    cuda::memcpy2DT(
        out.ptr(),  out.pitch(),
        m_g_sorted, m_pitchOut,
        out.cols(), 2,
        cudaMemcpyDeviceToHost );

    cuda::deviceSynchronize();
}







} //< namespace cudaNN
} //< namespace mpblocks


#endif // POINTSET_HPP_
