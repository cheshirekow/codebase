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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_cert/include/mpblocks/cuda_cert/CertSet.h
 *
 *  @date   Oct 26, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_CERT_CERTSET_H_
#define MPBLOCKS_CUDA_CERT_CERTSET_H_

#include <vector>
#include <cassert>
#include <mpblocks/cuda.h>
#include <mpblocks/cuda/linalg2.h>

namespace mpblocks  {
namespace cuda_cert {

typedef unsigned int uint_t;

class CertSet;

/// global offset of and size of vertex and face buffer for a particular
/// certificate volume
struct CertDef
{
    typedef unsigned int uint_t;
    uint_t  offV;
    uint_t  nV;
    uint_t  offF;
    uint_t  nF;

    CertDef(){}

    CertDef( uint_t offV, uint_t offF ):
        offV(offV),
        nV(0),
        offF(offF),
        nF(0)
    {}
};




/// provides a convenience interface for managing a set of workspace certificates
/// in GPU memory, and dispatching brute force CUDA searches on that point set
class CertSet
{
    public:
        uint_t    m_allocV;    ///< size of the vertex set allocated
        uint_t    m_allocF;    ///< size of the face   set allocated

        uint_t    m_nV;         ///< number of vertices so far
        uint_t    m_nF;         ///< number of faces so far

        uint_t    m_pitchV;     ///< row-pitch of buffers (in *bytes*)
        uint_t    m_pitchF;     ///< row-pitch of buffers (in *bytes*)

        float*    m_g_inV;      ///< kernel input buffer
        float*    m_g_inF;      ///< kernel input buffer
        int*      m_g_out;      ///< kernel output buffer
        uint_t    m_nOut;       ///< size of output buffer

        float*    m_g_dbg;      ///< debug output

        uint_t  m_threadsPerBlock;  ///< maximum threads per block
        uint_t  m_nSM;              ///< number of multiprocessors

        CertDef m_cert; ///< current cert being written
        std::vector<CertDef> m_certs;

    public:
        CertSet(uint_t nV=10, uint_t nF=10, uint_t nOut=100);
        ~CertSet();

        /// deallocate and zero out pointers
        void deallocate();

        /// reallocates device storage for a certificates set of size n with
        /// nV vertices and nF faces
        void allocate(uint_t nV, uint_t nF, uint_t nOut);

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

        /// compute the grid size given the current configuration and size of
        /// the point set
        void computeGrid( uint_t cert, uint_t& blocks, uint_t& threads );

    public:
        /// insert a vertex into the vertex set
        void insert_vertex( float v[3] );

        /// insert a face into the face set
        void insert_face( float f[4] );

        /// finish writing the current ceritficate volume
        void finish();

};


struct FinishTag{};
const FinishTag FINISH = FinishTag();

inline CertSet& operator<<( CertSet& set, FinishTag )
{
    set.finish();
    return set;
}

template < class ArrayLike >
struct VertexSurrogate
{
    const ArrayLike& v;
    VertexSurrogate( const ArrayLike& v ): v(v){}
};

template < class ArrayLike >
inline
VertexSurrogate<ArrayLike> vertex( const ArrayLike& v )
{
    return VertexSurrogate<ArrayLike>(v);
}

template < class ArrayLike, typename Scalar >
struct FaceSurrogate
{
    const ArrayLike& n;
    const Scalar&    d;

    FaceSurrogate( const ArrayLike& n, const Scalar& d ): n(n),d(d){}
};

template < class ArrayLike, class Scalar >
inline
FaceSurrogate<ArrayLike,Scalar> face( const ArrayLike& n, const Scalar& d )
{
    return FaceSurrogate<ArrayLike,Scalar>(n,d);
}

template < class ArrayLike >
inline CertSet& operator<<( CertSet& set, const VertexSurrogate<ArrayLike>& vs)
{
    float data[3];
    for(int i=0; i < 3; i++)
        data[i] = vs.v[i];
    set.insert_vertex(data);
    return set;
}

template < class ArrayLike, typename Scalar >
inline CertSet& operator<<( CertSet& set, const FaceSurrogate<ArrayLike,Scalar>& fs )
{
    float data[4];
    for(int i=0; i < 3; i++)
        data[i] = fs.n[i];
    data[3] = fs.d;
    set.insert_face(data);
    return set;
}



} //< namespace cuda_cert
} //< namespace mpblocks












#endif // CERTSET_H_
