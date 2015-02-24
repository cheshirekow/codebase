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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_cert/include/mpblocks/cuda_cert/dispatch.h
 *
 *  @date   Oct 26, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_CERT_DISPATCH_H_
#define MPBLOCKS_CUDA_CERT_DISPATCH_H_

#include <mpblocks/cuda_cert/CertSet.h>
#include <mpblocks/cuda_cert/CertSet2.h>
#include <mpblocks/cuda.h>
#include <mpblocks/cuda/linalg2.h>

namespace mpblocks  {
namespace cuda_cert {


/// check if the specified certificate is in collision
typedef cuda::linalg2::Matrix<float,3,3> Matrix3f;
typedef cuda::linalg2::Matrix<float,3,1> Vector3f;
int inContact( CertSet& certset, uint_t cert,
                    const Matrix3f& R0,
                    const Matrix3f& Rv,
                    const Vector3f& T0,
                    const Vector3f& dT,
                    float gamma,
                    float dilate);

int inContact_dbg( CertSet& certset, uint_t cert,
                    const Matrix3f& R0,
                    const Matrix3f& Rv,
                    const Vector3f& T0,
                    const Vector3f& dT,
                    float gamma,
                    float dilate);

/// check if the specified certificate is in collision
int inContact( CertSet2& certset, uint_t obj, uint_t cert,
                    const Matrix3f& R0,
                    const Matrix3f& Rv,
                    const Vector3f& T0,
                    const Vector3f& dT,
                    float gamma);

int inContact_dbg( CertSet2& certset, uint_t obj, uint_t cert,
                    const Matrix3f& R0,
                    const Matrix3f& Rv,
                    const Vector3f& T0,
                    const Vector3f& dT,
                    float gamma);



} //< namespace cuda_cert
} //< namespace mpblocks










#endif // DISPATCH_H_
