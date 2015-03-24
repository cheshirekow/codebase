/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of openbook.
 *
 *  openbook is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  openbook is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with openbook.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   dubins/curves_cuda/kernels.cu
 *
 *  @date   Jun 13, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_KERNELS_CU_HPP_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_KERNELS_CU_HPP_


namespace    mpblocks {
namespace      dubins {
namespace curves_cuda {
namespace     kernels {

namespace linalg = cuda::linalg2;




template< SolutionId Id, typename Format_t >
__device__  void applySolver(
        const linalg::Matrix<Format_t,3,1>& q0,
        const linalg::Matrix<Format_t,3,1>& q1,
        const Format_t r,
        DistanceAndId<Format_t>& best )
{
    Result<Format_t> soln = Solver<Id,Format_t>::solve(q0,q1,r);
    if( soln.f && soln.d < best.d )
    {
        best.d  = soln.d;
        best.id = Id;
    }
}

template< SolutionId Id, typename Format_t >
__device__  void applySolver(
        const linalg::Matrix<Format_t,3,1>& q0,
        const linalg::Matrix<Format_t,3,1>& q1,
        const Format_t r,
        Format_t& best )
{
    Result<Format_t> soln = Solver<Id,Format_t>::solve(q0,q1,r);
    if( soln.f && soln.d < best )
        best = soln.d;
}











template< typename Format_t >
__device__  void writeSolution(
        DebugCurved<Format_t>& soln,
        Result<Format_t>& result,
        int       off,
        int       pitch,
        int       idx,
        Format_t* g_out)
{
    // DebugCurved has 11 elements
    //      3 x 2ea center points
    //      3 x 1ea distances
    //      1       total distance
    //      1       feasible
    off *= 11;

    #pragma unroll
    for(int i=0; i < 3; i++)
    {
        #pragma unroll
        for(int j=0; j < 2; j++)
        {
            const int k = i*2 + j;
            __syncthreads();
            g_out[ (off + k)*pitch +idx ] = soln.c[i][j];
        }
    }

    #pragma unroll
    for(int i=0; i < 3; i++)
    {
        const int k = 6+i;
        __syncthreads();
        g_out[ (off + k)*pitch + idx ] = soln.l[i];
    }

    __syncthreads();
    g_out[ (off +  9)*pitch + idx ] = result.d;
    __syncthreads();
    g_out[ (off + 10)*pitch + idx ] = result.f ? 1 : 0;
}




template< typename Format_t >
__device__  void writeSolution(
        DebugStraight<Format_t>& soln,
        Result<Format_t>& result,
        int       off,
        int       pitch,
        int       idx,
        Format_t* g_out)
{
    // DebugStraight has 13 elements
    //      2 x 2ea center points
    //      2 x 2ea tangent points
    //      3 x 1ea distances
    //      1       total distance
    //      1       feasible
    // In addition, the DebugStraight block is written after the DebugCurved
    // block. g_out is moved to point to the DebugStraight block but the
    // first Enum value corresponding to a straight output is four, not zero
    off = (off - 4)*13;

    #pragma unroll
    for(int i=0; i < 2; i++)
    {
        #pragma unroll
        for(int j=0; j < 2; j++)
        {
            const int k = i*2 + j;
            __syncthreads();
            g_out[ (off + k)*pitch +idx ] = soln.c[i][j];
        }
    }

    #pragma unroll
    for(int i=0; i < 2; i++)
    {
        #pragma unroll
        for(int j=0; j < 2; j++)
        {
            const int k = 4 + i*2 + j;
            __syncthreads();
            g_out[ (off + k)*pitch +idx ] = soln.c[i][j];
        }
    }

    #pragma unroll
    for(int i=0; i < 3; i++)
    {
        const int k = 8+i;
        __syncthreads();
        g_out[ (off + k)*pitch + idx ] = soln.l[i];
    }

    __syncthreads();
    g_out[ (off + 11)*pitch + idx ] = result.d;
    __syncthreads();
    g_out[ (off + 12)*pitch + idx ] = result.f ? 1 : 0;
}












template< typename Format_t>
__global__  void distance_to_set(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       )
{
    using namespace linalg;

    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int N        = blockDim.x;  ///< number of blocks

    // which data point we work on
    int idx      = blockId * N + threadId;

    // if our idx is greater than the number of data points then we are a
    // left-over thread so just bail
    if( idx > n )
        return;

    // compose the query object
    linalg::Matrix<Format_t,3,1> q0,q1;
    set<0>( q0 ) = p.q[0];
    set<1>( q0 ) = p.q[1];
    set<2>( q0 ) = p.q[2];
    Format_t r  = p.r;

    // read in the target point q1, we synchronize between reads so that
    // reads are coallesced for maximum throughput
    set<0>(q1) = g_in[0*pitchIn + idx];
    __syncthreads();
    set<1>(q1) = g_in[1*pitchIn + idx];
    __syncthreads();
    set<2>(q1) = g_in[2*pitchIn + idx];
    __syncthreads();

    // best solution found
    Format_t    dBest;

    // now compute the distance for each of the solvers, and then record
    // the minimum one

    // the solutions with straight segments are always feasible so let's start
    // with one of them
    Result<Format_t> soln = Solver<LSL,Format_t>::solve(q0,q1,r);

    applySolver<LSR, Format_t>(q0,q1,r,dBest);
    applySolver<RSR, Format_t>(q0,q1,r,dBest);
    applySolver<RSL, Format_t>(q0,q1,r,dBest);
    applySolver<RLRa,Format_t>(q0,q1,r,dBest);
    applySolver<RLRb,Format_t>(q0,q1,r,dBest);
    applySolver<LRLa,Format_t>(q0,q1,r,dBest);
    applySolver<LRLb,Format_t>(q0,q1,r,dBest);

    __syncthreads();
    g_out[0*pitchOut + idx] = dBest;
}











template< typename Format_t >
__global__  void distance_from_set(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       )
{
    using namespace cuda::linalg2;

    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int N        = blockDim.x;  ///< number of blocks

    // which data point we work on
    int idx      = blockId * N + threadId;

    // if our idx is greater than the number of data points then we are a
    // left-over thread so just bail
    if( idx > n )
        return;

    // compose the query object
    linalg::Matrix<Format_t,3,1> q0,q1;
    set<0>( q0 ) = p.q[0];
    set<1>( q0 ) = p.q[1];
    set<2>( q0 ) = p.q[2];
    Format_t r  = p.r;

    // read in the target point q1, we synchronize between reads so that
    // reads are coallesced for maximum throughput
    set<0>(q1) = g_in[0*pitchIn + idx];
    __syncthreads();
    set<1>(q1) = g_in[1*pitchIn + idx];
    __syncthreads();
    set<2>(q1) = g_in[2*pitchIn + idx];
    __syncthreads();

    // best solution found
    Format_t    dBest;

    // now compute the distance for each of the solvers, and then record
    // the minimum one

    // the solutions with straight segments are always feasible so let's start
    // with one of them
    Result<Format_t> soln = Solver<LSL,Format_t>::solve(q1,q0,r);

    applySolver<LSR, Format_t>(q1,q0,r,dBest);
    applySolver<RSR, Format_t>(q1,q0,r,dBest);
    applySolver<RSL, Format_t>(q1,q0,r,dBest);
    applySolver<RLRa,Format_t>(q1,q0,r,dBest);
    applySolver<RLRb,Format_t>(q1,q0,r,dBest);
    applySolver<LRLa,Format_t>(q1,q0,r,dBest);
    applySolver<LRLb,Format_t>(q1,q0,r,dBest);

    __syncthreads();
    g_out[0*pitchOut + idx] = dBest;
}











template< typename Format_t>
__global__  void distance_to_set_with_id(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       )
{
    using namespace cuda::linalg2;

    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int N        = blockDim.x;  ///< number of blocks

    // which data point we work on
    int idx      = blockId * N + threadId;

    // if our idx is greater than the number of data points then we are a
    // left-over thread so just bail
    if( idx > n )
        return;

    // compose the query object
    linalg::Matrix<Format_t,3,1> q0,q1;
    set<0>( q0 ) = p.q[0];
    set<1>( q0 ) = p.q[1];
    set<2>( q0 ) = p.q[2];
    Format_t r  = p.r;

    // read in the target point q1, we synchronize between reads so that
    // reads are coallesced for maximum throughput
    set<0>(q1) = g_in[0*pitchIn + idx];
    __syncthreads();
    set<1>(q1) = g_in[1*pitchIn + idx];
    __syncthreads();
    set<2>(q1) = g_in[2*pitchIn + idx];
    __syncthreads();

    // now compute the distance for each of the solvers, and then record
    // the minimum one

    // the solutions with straight segments are always feasible so let's start
    // with one of them
    Result<Format_t> soln = Solver<LSL,Format_t>::solve(q0,q1,r);
    DistanceAndId<Format_t> dBest(soln.d,LSL);

    applySolver<LSR, Format_t>(q0,q1,r,dBest);
    applySolver<RSR, Format_t>(q0,q1,r,dBest);
    applySolver<RSL, Format_t>(q0,q1,r,dBest);
    applySolver<RLRa,Format_t>(q0,q1,r,dBest);
    applySolver<RLRb,Format_t>(q0,q1,r,dBest);
    applySolver<LRLa,Format_t>(q0,q1,r,dBest);
    applySolver<LRLb,Format_t>(q0,q1,r,dBest);

    typedef typename PackedStorage<sizeof(Format_t)>::Result Unsigned;
    Unsigned  pack = (idx << 4 ) | dBest.id;
    Unsigned* out  = reinterpret_cast<Unsigned*>(g_out + pitchOut);

    __syncthreads();
    g_out[idx] = dBest.d;
    __syncthreads();
    out[idx]   = pack;
}











template< typename Format_t >
__global__  void distance_from_set_with_id(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       )
{
    using namespace cuda::linalg2;

    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int N        = blockDim.x;  ///< number of blocks

    // which data point we work on
    int idx      = blockId * N + threadId;

    // if our idx is greater than the number of data points then we are a
    // left-over thread so just bail
    if( idx > n )
        return;

    // compose the query object
    linalg::Matrix<Format_t,3,1> q0,q1;
    set<0>( q0 ) = p.q[0];
    set<1>( q0 ) = p.q[1];
    set<2>( q0 ) = p.q[2];
    Format_t r  = p.r;

    // read in the target point q1, we synchronize between reads so that
    // reads are coallesced for maximum throughput
    set<0>(q1) = g_in[0*pitchIn + idx];
    __syncthreads();
    set<1>(q1) = g_in[1*pitchIn + idx];
    __syncthreads();
    set<2>(q1) = g_in[2*pitchIn + idx];
    __syncthreads();

    // now compute the distance for each of the solvers, and then record
    // the minimum one

    // the solutions with straight segments are always feasible so let's start
    // with one of them
    Result<Format_t> soln = Solver<LSL,Format_t>::solve(q0,q1,r);
    DistanceAndId<Format_t> dBest(soln.d,LSL);

    applySolver<LSR, Format_t>(q1,q0,r,dBest);
    applySolver<RSR, Format_t>(q1,q0,r,dBest);
    applySolver<RSL, Format_t>(q1,q0,r,dBest);
    applySolver<RLRa,Format_t>(q1,q0,r,dBest);
    applySolver<RLRb,Format_t>(q1,q0,r,dBest);
    applySolver<LRLa,Format_t>(q1,q0,r,dBest);
    applySolver<LRLb,Format_t>(q1,q0,r,dBest);

    typedef typename PackedStorage<sizeof(Format_t)>::Result Unsigned;
    Unsigned  pack = (idx << 4 ) | dBest.id;
    Unsigned* out  = reinterpret_cast<Unsigned*>(g_out + pitchOut);

    __syncthreads();
    g_out[idx] = dBest.d;
    __syncthreads();
    out[idx]   = pack;
}











template< typename Format_t>
__global__  void distance_to_set_debug(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       )
{
    using namespace cuda::linalg2;

    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int N        = blockDim.x;  ///< number of blocks

    // which data point we work on
    int idx      = blockId * N + threadId;

    // if our idx is greater than the number of data points then we are a
    // left-over thread so just bail
    if( idx > n )
        return;

    // compose the query object
    linalg::Matrix<Format_t,3,1> q0,q1;
    set<0>( q0 ) = p.q[0];
    set<1>( q0 ) = p.q[1];
    set<2>( q0 ) = p.q[2];
    Format_t r  = p.r;

    // read in the target point q1, we synchronize between reads so that
    // reads are coallesced for maximum throughput
    set<0>(q1) = g_in[0*pitchIn + idx];
    __syncthreads();
    set<1>(q1) = g_in[1*pitchIn + idx];
    __syncthreads();
    set<2>(q1) = g_in[2*pitchIn + idx];
    __syncthreads();

    // storage for the solution
    Result<Format_t> soln;

    {
        DebugCurved<Format_t> debug;

        // now compute the distance for each of the solve_debugrs
        soln = Solver<LRLa,Format_t>::solve_debug(q0,q1,r,debug);
        writeSolution(debug,soln,LRLa,pitchOut,idx,g_out);

        soln = Solver<LRLb,Format_t>::solve_debug(q0,q1,r,debug);
        writeSolution(debug,soln,LRLb,pitchOut,idx,g_out);

        soln = Solver<RLRa,Format_t>::solve_debug(q0,q1,r,debug);
        writeSolution(debug,soln,RLRa,pitchOut,idx,g_out);

        soln = Solver<RLRb,Format_t>::solve_debug(q0,q1,r,debug);
        writeSolution(debug,soln,RLRb,pitchOut,idx,g_out);
    }

    // there are 4 curved solutions, each one 11 elements, and pitchOut cols
    // so we advance the write head by
    g_out += 44*pitchOut;
    {
        DebugStraight<Format_t> debug;
        soln = Solver<LSL,Format_t>::solve_debug(q0,q1,r,debug);
        writeSolution(debug,soln,LSL,pitchOut,idx,g_out);

        soln = Solver<RSL,Format_t>::solve_debug(q0,q1,r,debug);
        writeSolution(debug,soln,RSL,pitchOut,idx,g_out);

        soln = Solver<RSR,Format_t>::solve_debug(q0,q1,r,debug);
        writeSolution(debug,soln,RSR,pitchOut,idx,g_out);

        soln = Solver<LSR,Format_t>::solve_debug(q0,q1,r,debug);
        writeSolution(debug,soln,LSR,pitchOut,idx,g_out);
    }


}











template< typename Format_t >
__global__  void distance_from_set_debug(
       Params<Format_t> p,  ///< radius and query state
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       )
{
    using namespace cuda::linalg2;

    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int N        = blockDim.x;  ///< number of blocks

    // which data point we work on
    int idx      = blockId * N + threadId;

    // if our idx is greater than the number of data points then we are a
    // left-over thread so just bail
    if( idx > n )
        return;

    // compose the query object
    linalg::Matrix<Format_t,3,1> q0,q1;
    set<0>( q0 ) = p.q[0];
    set<1>( q0 ) = p.q[1];
    set<2>( q0 ) = p.q[2];
    Format_t r  = p.r;

    // read in the target point q1, we synchronize between reads so that
    // reads are coallesced for maximum throughput
    set<0>(q1) = g_in[0*pitchIn + idx];
    __syncthreads();
    set<1>(q1) = g_in[1*pitchIn + idx];
    __syncthreads();
    set<2>(q1) = g_in[2*pitchIn + idx];
    __syncthreads();

    // storage for the solution
    Result<Format_t> soln;

    {
        DebugCurved<Format_t> debug;

        // now compute the distance for each of the solvers
        soln = Solver<LRLa,Format_t>::solve_debug(q1,q0,r,debug);
        writeSolution(debug,soln,LRLa,pitchOut,idx,g_out);

        soln = Solver<LRLb,Format_t>::solve_debug(q1,q0,r,debug);
        writeSolution(debug,soln,LRLb,pitchOut,idx,g_out);

        soln = Solver<RLRa,Format_t>::solve_debug(q1,q0,r,debug);
        writeSolution(debug,soln,RLRa,pitchOut,idx,g_out);

        soln = Solver<RLRb,Format_t>::solve_debug(q1,q0,r,debug);
        writeSolution(debug,soln,RLRb,pitchOut,idx,g_out);
    }

    // there are 4 curved solutions, each one 11 elements, and pitchOut cols
    // so we advance the write head by
    g_out += 44*pitchOut;
    {
        DebugStraight<Format_t> debug;
        soln = Solver<LSL,Format_t>::solve_debug(q1,q0,r,debug);
        writeSolution(debug,soln,LSL,pitchOut,idx,g_out);

        soln = Solver<RSL,Format_t>::solve_debug(q1,q0,r,debug);
        writeSolution(debug,soln,RSL,pitchOut,idx,g_out);

        soln = Solver<RSR,Format_t>::solve_debug(q1,q0,r,debug);
        writeSolution(debug,soln,RSR,pitchOut,idx,g_out);

        soln = Solver<LSR,Format_t>::solve_debug(q1,q0,r,debug);
        writeSolution(debug,soln,LSR,pitchOut,idx,g_out);
    }
}











template< typename Format_t >
__global__  void group_distance_to_set(
       EuclideanParams<Format_t> p,
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       )
{
    using namespace cuda::linalg2;

    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int N        = blockDim.x;  ///< number of blocks

    // which data point we work on
    int idx      = blockId * N + threadId;

    // if our idx is greater than the number of data points then we are a
    // left-over thread so just bail
    if( idx > n )
        return;

    // compose the query object
    Matrix<Format_t,3,1> q0, q1, diff;
    set<0>( q0 ) = p.q[0];
    set<1>( q0 ) = p.q[1];
    set<2>( q0 ) = p.q[2];

    // read in the target point q1, we synchronize between reads so that
    // reads are coallesced for maximum throughput
    set<0>( q1 ) = g_in[0*pitchIn + idx];
    __syncthreads();
    set<1>( q1 ) = g_in[1*pitchIn + idx];
    __syncthreads();
    set<2>( q1 ) = g_in[2*pitchIn + idx];
    __syncthreads();

    // find the difference
    diff = q1 - q0;

    // correct the rotation
    const Format_t _PI = static_cast<Format_t>(M_PI);
    if( get<2>( diff ) > _PI )
        set<2>( diff ) -= 2*_PI;
    if( get<2>( diff ) < _PI )
        set<2>( diff ) += 2*_PI;

    Format_t dist2 = norm_squared(diff);

    __syncthreads();
    g_out[0*pitchOut + idx] = dist2;
}











template< typename Format_t>
__global__  void group_distance_to_set_with_id(
       EuclideanParams<Format_t> p,
       Format_t*    g_in,
       unsigned int pitchIn,
       Format_t*    g_out,
       unsigned int pitchOut,
       unsigned int n
       )
{
    using namespace cuda::linalg2;

    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int N        = blockDim.x;  ///< number of blocks

    // which data point we work on
    int idx      = blockId * N + threadId;

    // if our idx is greater than the number of data points then we are a
    // left-over thread so just bail
    if( idx > n )
        return;

    // compose the query object
    Matrix<Format_t,3,1> q0, q1, diff;
    set<0>( q0 ) = p.q[0];
    set<1>( q0 ) = p.q[1];
    set<2>( q0 ) = p.q[2];

    // read in the target point q1, we synchronize between reads so that
    // reads are coallesced for maximum throughput
    set<0>( q1 ) = g_in[0*pitchIn + idx];
    __syncthreads();
    set<1>( q1 ) = g_in[1*pitchIn + idx];
    __syncthreads();
    set<2>( q1 ) = g_in[2*pitchIn + idx];
    __syncthreads();

    // find the difference
    diff = q1 - q0;

    // correct the rotation
    const Format_t _PI = static_cast<Format_t>(M_PI);
    if( get<2>( diff ) > _PI )
        set<2>( diff ) -= 2*_PI;
    if( get<2>( diff ) < _PI )
        set<2>( diff ) += 2*_PI;

    Format_t dist2 = norm_squared(diff);

    typedef typename PackedStorage<sizeof(Format_t)>::Result Unsigned;
    Unsigned* out  = reinterpret_cast<Unsigned*>(g_out + pitchOut);

    __syncthreads();
    g_out[idx] = dist2;
    __syncthreads();
    out[idx]   = idx;
}





} // kernels
} // curves
} // dubins
} // mpblocks


#endif

