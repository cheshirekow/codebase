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

#include <iostream>
#include <mpblocks/cuda.hpp>
#include <mpblocks/cuda_cert/CertSet.h>
#include <mpblocks/cuda_cert/debug.h>


namespace mpblocks  {
namespace cuda_cert {








CertSet::CertSet(uint_t nV, uint_t nF, uint_t nOut)
{
    m_g_inV   = 0;
    m_g_inF   = 0;
    m_g_out   = 0;
    m_g_dbg   = 0;

    m_threadsPerBlock = 0;
    m_nSM             = 0;

    deallocate();
    clear();

    try
    {
        config();
        allocate(nV,nF,nOut);
    }
    catch( const std::exception& ex )
    {
        std::cerr << "Error in constructing dubins CUDA PointSet: "
                  << ex.what()
                  << "\nNote: point set is unallocated\n";
    }
}



CertSet::~CertSet()
{
    deallocate();
}











void CertSet::deallocate()
{
    if(m_g_inV)
    {
        cuda::free(m_g_inV);
        m_g_inV = 0;
    }

    if(m_g_inF)
    {
        cuda::free(m_g_inF);
        m_g_inF = 0;
    }

    if(m_g_out)
    {
        cuda::free(m_g_out);
        m_g_out = 0;
    }

    if(m_g_dbg)
    {
        cuda::free(m_g_dbg);
        m_g_dbg = 0;
    }

    m_allocV = 0;
    m_allocF = 0;
}











void CertSet::allocate(uint_t nV, uint_t nF, uint_t nOut)
{
    deallocate();

    m_allocV    = nV;
    m_allocF    = nF;

    size_t pitchV;
    m_g_inV  = cuda::mallocPitchT<float>( pitchV, m_allocV, 3 );
    m_pitchV = pitchV;
    std::cout << "allocated m_g_in for "
                  << m_allocV << " object with pitch: " << m_pitchV << "\n";

    size_t pitchF;
    m_g_inF  = cuda::mallocPitchT<float>( pitchF, m_allocF, 4 );
    m_pitchF = pitchF;
    std::cout << "allocated m_g_sorted for "
                  << m_allocF << " object with pitch: " << m_pitchF << "\n";

    m_g_out = cuda::mallocT<int>(nOut);
    m_nOut = nOut;
    std::cout << "allocated m_g_out for " << nOut << " floats\n";

    m_g_dbg = cuda::mallocT<float>(sizeDebugBuffer);
    std::cout << "allocated m_g_out dbg " << nOut << " floats\n";
}



















void CertSet::clear(bool clearmem)
{
    m_nV = 0;
    m_nF = 0;
    m_certs.clear();
    m_cert = CertDef(0,0);
    if( clearmem )
    {
        cuda::memset2DT( m_g_inV,    m_pitchV,  0, m_allocV,  3 );
        cuda::memset2DT( m_g_inF,    m_pitchF,  0, m_allocF,  4 );
        cuda::memset( m_g_out, 0, m_nOut*sizeof(int) );
        cuda::memset( m_g_dbg, 0, sizeDebugBuffer*sizeof(float) );
    }
}











void CertSet::config()
{
    int devId = cuda::getDevice();
    config(devId);
}










void CertSet::computeGrid( uint_t cert, uint_t& blocks, uint_t& threads )
{
    CertDef def = m_certs[cert];
    uint_t totalWork = def.nF*def.nV;

    threads = cuda::intDivideRoundUp(totalWork,m_nSM);
    if( threads > m_threadsPerBlock )
        threads = m_threadsPerBlock;
    blocks  = cuda::intDivideRoundUp(totalWork,threads);
}







void CertSet::insert_vertex( float v[3] )
{
    cuda::memcpy2DT( m_g_inV + m_nV, m_pitchV,
                     v, sizeof(float),
                     1, 3,
                     cudaMemcpyHostToDevice );
    m_nV++;
    m_cert.nV++;
}

void CertSet::insert_face( float f[5] )
{
    cuda::memcpy2DT( m_g_inF + m_nF, m_pitchF,
                     f, sizeof(float),
                     1, 4,
                     cudaMemcpyHostToDevice );
    m_nF++;
    m_cert.nF++;
}

void CertSet::finish()
{
    m_certs.push_back(m_cert);
    m_cert = CertDef(m_nV,m_nF);
}







} //< namespace cuda_cert
} //< namespace mpblocks











