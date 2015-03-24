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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_cert/test/kernel_props.cpp
 *
 *  @date   Oct 26, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <mpblocks/cuda.h>
#include <mpblocks/cuda_cert/fattr.h>
#include <mpblocks/cuda_cert/CertSet.h>
#include <mpblocks/cuda_cert/dispatch.h>


using namespace mpblocks;

int main(int argc, char** argv)
{
    cuda_cert::fattrMap_t map;
    cuda_cert::get_fattr(map);

    std::cout << "Kernel attributes:\n-----------------"
              << "\n  binary version : " << map["check_cert"].binaryVersion
              << "\n     ptx version : " << map["check_cert"].ptxVersion
              << "\n      const size : " << map["check_cert"].constSizeBytes
              << "\n      local size : " << map["check_cert"].localSizeBytes
              << "\n     shared size : " << map["check_cert"].sharedSizeBytes
              << "\n  num 32bit regs : " << map["check_cert"].numRegs
              << "\n  threads / block: " << map["check_cert"].maxThreadsPerBlock
              << "\n";

    cuda_cert::CertSet set(100,100,100);

    Eigen::Vector3f v;
    Eigen::Vector3f n;

    for(int j=0; j < 10; j++)
    {
        using namespace cuda_cert;
        int nV = 2 + rand()%10;
        int nF = 2 + rand()%10;

        for(int i=0; i < nV; i++)
        {
            v << (rand() % 100)/100.0f,
                 (rand() % 100)/100.0f,
                 (rand() % 100)/100.0f;
            set << vertex(v);
        }

        for(int i=0; i < nF; i++)
        {
            n << (rand() % 100)/100.0f,
                 (rand() % 100)/100.0f,
                 (rand() % 100)/100.0f;
            float d = (rand() % 100)/100.0f;
            set << face(n,d);
        }
        set << FINISH;
    }

    using namespace cuda::linalg2;
    Matrix<float,3,3> R0;
    R0 << 1, 0, 0,
          0, 1, 0,
          0, 0, 1;

    Matrix<float,3,3> Rv;
    Rv << 1, 0, 0,
          0, 1, 0,
          0, 0, 1;
    Matrix<float,3,1> T0,dT;
    T0 << 0, 0, 0;
    dT << 1, 1, 1;

    bool result = cuda_cert::inContact(set,0,R0,Rv,T0,dT,0.5f,1.0f);
    std::cout << "result : " << (result ? "true" : "false" ) << "\n";

    return 0;
}




