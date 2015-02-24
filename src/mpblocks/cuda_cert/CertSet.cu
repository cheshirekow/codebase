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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_cert/src/CertSet.cpp
 *
 *  @date   Oct 26, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <cstdio>
#include <mpblocks/cuda_cert/CertSet.h>

#include <mpblocks/cuda.hpp>
#include <mpblocks/cuda_cert/kernels.cu.h>
#include <mpblocks/cuda_cert/dispatch.h>
#include <mpblocks/cuda_cert/debug.h>


namespace mpblocks  {
namespace cuda_cert {



void CertSet::config(int devId)
{
    cuda::DeviceProp     devProps(devId);
    cuda::FuncAttributes attr;
    uint_t  maxRegs     = 0;

    attr.getFrom( &kernels::check_cert );
    maxRegs     = std::max(maxRegs,   (uint_t)attr.numRegs);

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
}




int inContact( CertSet& set, uint_t cert,
                            const Matrix3f& R0,
                            const Matrix3f& Rv,
                            const Vector3f& T0,
                            const Vector3f& dT,
                            float gamma,
                            float dilate  )
{
    uint_t blocks,threads;
    set.computeGrid(cert,blocks,threads);

    size_t pitchV  = set.m_pitchV/sizeof(float);
    size_t pitchF  = set.m_pitchF/sizeof(float);

    CertDef def = set.m_certs[cert];

//    std::cout << "calling kernel for cert (vOff,nV,fOff,nF) : "
//              << def.offV << ", "
//              << def.nV   << ", "
//              << def.offF << ", "
//              << def.nF << "\n";

    // call the kernel
    kernels::check_cert<<<blocks,threads>>>(
        set.m_g_inV,
        def.offV,
        def.nV,
        pitchV,
        set.m_g_inF,
        def.offF,
        def.nF,
        pitchF,
        R0,
        Rv,
        T0,
        dT,
        gamma,
        dilate,
        set.m_g_out
        );
    cuda::deviceSynchronize();

    // retrieve results
    std::vector<int> results(def.nV*def.nF);
    cuda::memcpyT<int>(
            results.data(),
            set.m_g_out,
            results.size(),
            cudaMemcpyDeviceToHost );

    for(int i=0; i < results.size(); i++)
        if( results[i] )
            return i;

    return -1;
}








int inContact_dbg( CertSet& set, uint_t cert,
                            const Matrix3f& R0,
                            const Matrix3f& Rv,
                            const Vector3f& T0,
                            const Vector3f& dT,
                            float gamma,
                            float dilate)
{
    uint_t blocks,threads;
    set.computeGrid(cert,blocks,threads);

    size_t pitchV  = set.m_pitchV/sizeof(float);
    size_t pitchF  = set.m_pitchF/sizeof(float);

    CertDef def = set.m_certs[cert];

//    std::cout << "calling kernel for cert (vOff,nV,fOff,nF) : "
//              << def.offV << ", "
//              << def.nV   << ", "
//              << def.offF << ", "
//              << def.nF << "\n";

    // call the kernel
    kernels::check_cert_dbg<<<blocks,threads>>>(
        set.m_g_inV,
        def.offV,
        def.nV,
        pitchV,
        set.m_g_inF,
        def.offF,
        def.nF,
        pitchF,
        R0,
        Rv,
        T0,
        dT,
        gamma,
        dilate,
        set.m_g_out,
        set.m_g_dbg
        );
    cuda::deviceSynchronize();

    // retrieve results
    std::vector<int> results(def.nV*def.nF);
    cuda::memcpyT<int>(
            results.data(),
            set.m_g_out,
            results.size(),
            cudaMemcpyDeviceToHost );

//    std::cout << "num zero crossings: ";
//    for(int i=0; i < results.size(); i++)
//        std::cout << results[i] << ", ";
//    std::cout << "\n";
//

    std::vector<float> debug(host_sizeDebugOutput);
    cuda::memcpyT<float>(
            debug.data(),
            set.m_g_dbg,
            debug.size(),
            cudaMemcpyDeviceToHost );
    std::vector<double> ddbg(host_sizeDebugOutput);
    for( int i=0; i < debug.size(); i++)
        ddbg[i] = debug[i];

    int idx=0;
    std::cout << "F cuda, thread 0:\n";
    int size = ddbg.size();
    for(int r=0; r < 3; r++)
    {
        for(int c = 0; c < 3; c++)
        {
            printf("%8.4f + %8.4fs + %8.4fs^2  ",
                    ddbg[idx+0], ddbg[idx+1],ddbg[idx+2]);
            idx += 3;
        }
        printf("%8.4f + %8.4fs + %8.4fs^2 + %8.4fs^3  \n",
                ddbg[idx+0], ddbg[idx+1],ddbg[idx+2],ddbg[idx+3]);
        idx+=4;
    }

    std::cout << "Fx cuda, thread 0:\n";
    for(int r=0; r < 4; r++)
    {
        printf("%8.4f + %8.4fs + %8.4fs^2 + %8.4fs^3  \n",
                 ddbg[idx+0], ddbg[idx+1],ddbg[idx+2],ddbg[idx+3]);
        idx+=4;
    }

    printf("\n v = (%8.4f,%8.4f,%8.4f) \n",
            ddbg[idx+0], ddbg[idx+1],ddbg[idx+2]);
    idx += 3;
    printf("(n,d) = (%8.4f,%8.4f,%8.4f) %8.4f\n",
            ddbg[idx+0], ddbg[idx+1],ddbg[idx+2],ddbg[idx+3]);
    idx+=4;
    printf("(nh,dh) = (%8.4f,%8.4f,%8.4f) %8.4f\n",
            ddbg[idx+0], ddbg[idx+1],ddbg[idx+2],ddbg[idx+3]);
    idx+=4;
    printf(" n' F x = %8.4f + %8.4fs + %8.4fs^2 + %8.4fs^3\n\n",
            ddbg[idx+0], ddbg[idx+1],ddbg[idx+2],ddbg[idx+3]);
    idx+=4;

    std::cout << std::endl;

    std::cout << "zero crossings: \n";
    for(int i=0; i < results.size(); i++)
        std::cout << "   " << i << " : " << results[i] << "\n";
    std::cout << "\n";

    for(int i=0; i < results.size(); i++)
        if( results[i] )
            return i;

    return -1;
}




} //< namespace cuda_cert
} //< namespace mpblocks










