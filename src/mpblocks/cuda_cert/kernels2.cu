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


#include <mpblocks/cuda_cert/kernels.cu.h>
#include <mpblocks/cuda_cert/debug.h>
#include <mpblocks/cuda.hpp>
#include <mpblocks/cuda/linalg2.h>
#include <mpblocks/cuda/polynomial.h>
#include <mpblocks/cuda/polynomial/SturmSequence3.h>



namespace    mpblocks {
namespace   cuda_cert {
namespace     kernels {


template< int v00,
              int v01=INT_MAX, int v02=INT_MAX, int v03=INT_MAX,
              int v04=INT_MAX, int v05=INT_MAX, int v06=INT_MAX,
              int v07=INT_MAX, int v08=INT_MAX, int v09=INT_MAX,
              int v10=INT_MAX, int v11=INT_MAX, int v12=INT_MAX>
struct p
{
    typedef cuda::polynomial::Polynomial<float, typename
            cuda::polynomial::intlist::construct<
            v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v10,v11,v12>::result > type;
};


__global__  void check_cert2_dbg(
       float* g_dataV,
       uint_t i0V,
       uint_t nV,
       uint_t pitchV,
       float* g_dataF,
       uint_t i0F,
       uint_t nF,
       uint_t pitchF,
       Matrix3f R0,
       Matrix3f Rv,
       Vector3f T0,
       Vector3f dT,
       float    gamma,
       int*     g_out
       ,float*   g_dbg)
{
//    __shared__ int hasContact;

    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int N        = blockDim.x;  ///< number of blocks

    // if we're the first thread in this block then initialize the block-vote
//    if( threadId == 0 )
//        hasContact = 0;
//    __syncthreads();

    // which vertex/face pair we're working on
    int idx      = blockId * N + threadId;

    // write everything out for debugging
    g_dbg += idx*sizeDebugOutput;

    // which vertex we're working on
    int iV = idx / nF;
    int iF = idx - (iV*nF);

    // if our idx is greater than the number of data points then we are a
    // left-over thread so just bail
    // @todo is this OK with non-power of
    // two array sizes and the fact that we syncthreads after this point?
    if( idx > nV*nF )
        return;

    // read in the vertex
    Vector3f v;
    set<0>(v) = g_dataV[ (i0V + iV) + pitchV*0 ];
    __syncthreads();
    set<1>(v) = g_dataV[ (i0V + iV) + pitchV*1 ];
    __syncthreads();
    set<2>(v) = g_dataV[ (i0V + iV) + pitchV*2 ];
    __syncthreads();

    // read in the face
    Vector3f n0;
    set<0>(n0) = g_dataF[ (i0F + iF) + pitchF*0 ];
    __syncthreads();
    set<1>(n0) = g_dataF[ (i0F + iF) + pitchF*1 ];
    __syncthreads();
    set<2>(n0) = g_dataF[ (i0F + iF) + pitchF*2 ];
    __syncthreads();

    float d0 = g_dataF[ (i0F + iF) + pitchF*3 ];

    __syncthreads();

    // compute the polynomial for interference checking
    using namespace cuda::polynomial::device_coefficient_key;
    using namespace cuda::polynomial::device_param_key;
    p<0,2>::type   gs2p1 = (1.0f + (gamma*gamma) * (s^_2) );

    p<2>::type   R_11 = (-gamma*gamma) * (s^_2);
    p<1>::type   R_12 = (-2*gamma) * (s^_1);
    p<1>::type   R_21 = (2*gamma)  * (s^_1);
    p<2>::type   R_22 = (-gamma*gamma) * (s^_2);

    // ugly manual matrix multiplication... goddamnit cuda why cant you just
    // intline the damn templates
    p<0,2>::type F_00 = get<0,0>(Rv)*gs2p1;
    p<0,2>::type F_01 = get<0,1>(Rv)*gs2p1;
    p<0,2>::type F_02 = get<0,2>(Rv)*gs2p1;

    p<1,2>::type   F_10 = R_11 * get<1,0>(Rv) +
                          R_12 * get<2,0>(Rv);
    p<1,2>::type   F_11 = R_11 * get<1,1>(Rv) +
                          R_12 * get<2,1>(Rv);
    p<1,2>::type   F_12 = R_11 * get<1,2>(Rv) +
                          R_12 * get<2,2>(Rv);

    p<1,2>::type   F_20 = R_21 * get<1,0>(Rv) +
                          R_22 * get<2,0>(Rv);
    p<1,2>::type   F_21 = R_21 * get<1,1>(Rv) +
                          R_22 * get<2,1>(Rv);
    p<1,2>::type   F_22 = R_21 * get<1,2>(Rv) +
                          R_22 * get<2,2>(Rv);

    p<0,1,2,3>::type F_03 =  gs2p1 * ( get<0>(T0) + get<0>(dT) * (s^_1) );
    p<0,1,2,3>::type F_13 =  gs2p1 * ( get<1>(T0) + get<1>(dT) * (s^_1) );
    p<0,1,2,3>::type F_23 =  gs2p1 * ( get<2>(T0) + get<2>(dT) * (s^_1) );

    // get the column vector which is F*_x_
    p<0,1,2,3>::type Fx_0 =     F_00 * get<0>(v)
                              + F_01 * get<1>(v)
                              + F_02 * get<2>(v)
                              + F_03 * 1.0f;
    p<0,1,2,3>::type Fx_1 =     F_10 * get<0>(v)
                              + F_11 * get<1>(v)
                              + F_12 * get<2>(v)
                              + F_13 * 1.0f;
    p<0,1,2,3>::type Fx_2 =     F_20 * get<0>(v)
                              + F_21 * get<1>(v)
                              + F_22 * get<2>(v)
                              + F_23 * 1.0f;
    p<0,1,2,3>::type Fx_3 =  gs2p1;

    if( idx < numDebugThreads )
    {
        *(g_dbg++) = get<0>(F_00);
        *(g_dbg++) = get<1>(F_00);
        *(g_dbg++) = get<2>(F_00);

        *(g_dbg++) = get<0>(F_01);
        *(g_dbg++) = get<1>(F_01);
        *(g_dbg++) = get<2>(F_01);

        *(g_dbg++) = get<0>(F_02);
        *(g_dbg++) = get<1>(F_02);
        *(g_dbg++) = get<2>(F_02);

        *(g_dbg++) = get<0>(F_03);
        *(g_dbg++) = get<1>(F_03);
        *(g_dbg++) = get<2>(F_03);
        *(g_dbg++) = get<3>(F_03);

        *(g_dbg++) = 0;
        *(g_dbg++) = get<1>(F_10);
        *(g_dbg++) = get<2>(F_10);

        *(g_dbg++) = 0;
        *(g_dbg++) = get<1>(F_11);
        *(g_dbg++) = get<2>(F_11);

        *(g_dbg++) = 0;
        *(g_dbg++) = get<1>(F_12);
        *(g_dbg++) = get<2>(F_12);

        *(g_dbg++) = get<0>(F_13);
        *(g_dbg++) = get<1>(F_13);
        *(g_dbg++) = get<2>(F_13);
        *(g_dbg++) = get<3>(F_13);

        *(g_dbg++) = 0;
        *(g_dbg++) = get<1>(F_20);
        *(g_dbg++) = get<2>(F_20);

        *(g_dbg++) = 0;
        *(g_dbg++) = get<1>(F_21);
        *(g_dbg++) = get<2>(F_21);

        *(g_dbg++) = 0;
        *(g_dbg++) = get<1>(F_22);
        *(g_dbg++) = get<2>(F_22);

        *(g_dbg++) = get<0>(F_23);
        *(g_dbg++) = get<1>(F_23);
        *(g_dbg++) = get<2>(F_23);
        *(g_dbg++) = get<3>(F_23);


        *(g_dbg++) = get<0>(Fx_0);
        *(g_dbg++) = get<1>(Fx_0);
        *(g_dbg++) = get<2>(Fx_0);
        *(g_dbg++) = get<3>(Fx_0);

        *(g_dbg++) = get<0>(Fx_1);
        *(g_dbg++) = get<1>(Fx_1);
        *(g_dbg++) = get<2>(Fx_1);
        *(g_dbg++) = get<3>(Fx_1);

        *(g_dbg++) = get<0>(Fx_2);
        *(g_dbg++) = get<1>(Fx_2);
        *(g_dbg++) = get<2>(Fx_2);
        *(g_dbg++) = get<3>(Fx_2);

        *(g_dbg++) = get<0>(Fx_3);
        *(g_dbg++) = get<1>(Fx_3);
        *(g_dbg++) = get<2>(Fx_3);
        *(g_dbg++) = get<3>(Fx_3);
    }

    // rotate the face to the start orientation
    Vector3f n = Rv*n0;
    float    d = d0;

    // now get n^T * F * _x_
    p<0,1,2,3>::type nFx = get<0>(n)*Fx_0
                        +  get<1>(n)*Fx_1
                        +  get<2>(n)*Fx_2
                        -          d*Fx_3;

    if( idx < numDebugThreads )
    {
        *(g_dbg++) = get<0>(v);
        *(g_dbg++) = get<1>(v);
        *(g_dbg++) = get<2>(v);

        *(g_dbg++) = get<0>(n0);
        *(g_dbg++) = get<1>(n0);
        *(g_dbg++) = get<2>(n0);
        *(g_dbg++) = d0;

        *(g_dbg++) = get<0>(n);
        *(g_dbg++) = get<1>(n);
        *(g_dbg++) = get<2>(n);
        *(g_dbg++) = -d;

        *(g_dbg++) = get<0>(nFx);
        *(g_dbg++) = get<1>(nFx);
        *(g_dbg++) = get<2>(nFx);
        *(g_dbg++) = get<3>(nFx);
    }

    // count the number of zero crossings
    g_out[idx] = cuda::polynomial::CountZeros<3>::count_zeros(nFx,0.0f,1.0f);

    // blockvote, writer is undefined but it will happen
//    if( numZeros )
//        hasContact = 1;

//    // wait for the blockvote to finish
//    __syncthreads();

//    // if we're the first thread in this block write the output
//    if( threadIdx.x == 0 )
//        g_out[blockIdx.x] = hasContact;
}



__global__  void check_cert2(
       float* g_dataV,
       uint_t i0V,
       uint_t nV,
       uint_t pitchV,
       float* g_dataF,
       uint_t i0F,
       uint_t nF,
       uint_t pitchF,
       Matrix3f R0,
       Matrix3f Rv,
       Vector3f T0,
       Vector3f dT,
       float    gamma,
       int*     g_out)
{
//    __shared__ int hasContact;

    int threadId = threadIdx.x;
    int blockId  = blockIdx.x;
    int N        = blockDim.x;  ///< number of blocks

    // if we're the first thread in this block then initialize the block-vote
//    if( threadId == 0 )
//        hasContact = 0;
//    __syncthreads();

    // which vertex/face pair we're working on
    int idx      = blockId * N + threadId;

    // which vertex we're working on
    int iV = idx / nF;
    int iF = idx - (iV*nF);

    // if our idx is greater than the number of data points then we are a
    // left-over thread so just bail
    // @todo is this OK with non-power of
    // two array sizes and the fact that we syncthreads after this point?
    if( idx > nV*nF )
        return;

    // read in the vertex
    Vector3f v;
    set<0>(v) = g_dataV[ (i0V + iV) + pitchV*0 ];
    __syncthreads();
    set<1>(v) = g_dataV[ (i0V + iV) + pitchV*1 ];
    __syncthreads();
    set<2>(v) = g_dataV[ (i0V + iV) + pitchV*2 ];
    __syncthreads();

    // read in the face
    Vector3f n0;
    set<0>(n0) = g_dataF[ (i0F + iF) + pitchF*0 ];
    __syncthreads();
    set<1>(n0) = g_dataF[ (i0F + iF) + pitchF*1 ];
    __syncthreads();
    set<2>(n0) = g_dataF[ (i0F + iF) + pitchF*2 ];
    __syncthreads();

    float d0 = g_dataF[ (i0F + iF) + pitchF*3 ];

    __syncthreads();

    // compute the polynomial for interference checking
    using namespace cuda::polynomial::device_coefficient_key;
    using namespace cuda::polynomial::device_param_key;
    p<0,2>::type   gs2p1 = (1.0f + (gamma*gamma) * (s^_2) );

    p<2>::type   R_11 = (-gamma*gamma) * (s^_2);
    p<1>::type   R_12 = (-2*gamma) * (s^_1);
    p<1>::type   R_21 = (2*gamma)  * (s^_1);
    p<2>::type   R_22 = (-gamma*gamma) * (s^_2);

    // ugly manual matrix multiplication... goddamnit cuda why cant you just
    // intline the damn templates
    p<0,2>::type F_00 = get<0,0>(Rv)*gs2p1;
    p<0,2>::type F_01 = get<0,1>(Rv)*gs2p1;
    p<0,2>::type F_02 = get<0,2>(Rv)*gs2p1;

    p<1,2>::type   F_10 = R_11 * get<1,0>(Rv) +
                          R_12 * get<2,0>(Rv);
    p<1,2>::type   F_11 = R_11 * get<1,1>(Rv) +
                          R_12 * get<2,1>(Rv);
    p<1,2>::type   F_12 = R_11 * get<1,2>(Rv) +
                          R_12 * get<2,2>(Rv);

    p<1,2>::type   F_20 = R_21 * get<1,0>(Rv) +
                          R_22 * get<2,0>(Rv);
    p<1,2>::type   F_21 = R_21 * get<1,1>(Rv) +
                          R_22 * get<2,1>(Rv);
    p<1,2>::type   F_22 = R_21 * get<1,2>(Rv) +
                          R_22 * get<2,2>(Rv);

    p<0,1,2,3>::type F_03 =  gs2p1 * ( get<0>(T0) + get<0>(dT) * (s^_1) );
    p<0,1,2,3>::type F_13 =  gs2p1 * ( get<1>(T0) + get<1>(dT) * (s^_1) );
    p<0,1,2,3>::type F_23 =  gs2p1 * ( get<2>(T0) + get<2>(dT) * (s^_1) );

    // get the column vector which is F*_x_
    p<0,1,2,3>::type Fx_0 =     F_00 * get<0>(v)
                              + F_01 * get<1>(v)
                              + F_02 * get<2>(v)
                              + F_03 * 1.0f;
    p<0,1,2,3>::type Fx_1 =     F_10 * get<0>(v)
                              + F_11 * get<1>(v)
                              + F_12 * get<2>(v)
                              + F_13 * 1.0f;
    p<0,1,2,3>::type Fx_2 =     F_20 * get<0>(v)
                              + F_21 * get<1>(v)
                              + F_22 * get<2>(v)
                              + F_23 * 1.0f;
    p<0,1,2,3>::type Fx_3 =  gs2p1;


    // rotate the face to the start orientation
    Vector3f n = Rv*n0;
    float d    = d0 ;

    // now get n^T * F * _x_
    p<0,1,2,3>::type nFx = get<0>(n)*Fx_0
                        +  get<1>(n)*Fx_1
                        +  get<2>(n)*Fx_2
                        -          d*Fx_3;

    // count the number of zero crossings
    g_out[idx] = cuda::polynomial::CountZeros<3>::count_zeros(nFx,0.0f,1.0f);

    // blockvote, writer is undefined but it will happen
//    if( numZeros )
//        hasContact = 1;

//    // wait for the blockvote to finish
//    __syncthreads();

//    // if we're the first thread in this block write the output
//    if( threadIdx.x == 0 )
//        g_out[blockIdx.x] = hasContact;
}














} // kernels
} // cudaNN
} // mpblocks



