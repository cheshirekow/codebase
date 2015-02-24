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
#include <mpblocks/cuda.h>
#include <mpblocks/cuda_cert/fattr.h>

int main(int argc, char** argv)
{
    mpblocks::cuda_cert::fattrMap_t map;
    mpblocks::cuda_cert::get_fattr(map);


    std::cout << "Kernel attributes (check_cert):\n-----------------"
              << "\n  binary version : " << map["check_cert"].binaryVersion
              << "\n     ptx version : " << map["check_cert"].ptxVersion
              << "\n      const size : " << map["check_cert"].constSizeBytes
              << "\n      local size : " << map["check_cert"].localSizeBytes
              << "\n     shared size : " << map["check_cert"].sharedSizeBytes
              << "\n  num 32bit regs : " << map["check_cert"].numRegs
              << "\n  threads / block: " << map["check_cert"].maxThreadsPerBlock
              << "\n";

    std::cout << "Kernel attributes (check_cert_dbg):\n-----------------"
              << "\n  binary version : " << map["check_cert_dbg"].binaryVersion
              << "\n     ptx version : " << map["check_cert_dbg"].ptxVersion
              << "\n      const size : " << map["check_cert_dbg"].constSizeBytes
              << "\n      local size : " << map["check_cert_dbg"].localSizeBytes
              << "\n     shared size : " << map["check_cert_dbg"].sharedSizeBytes
              << "\n  num 32bit regs : " << map["check_cert_dbg"].numRegs
              << "\n  threads / block: " << map["check_cert_dbg"].maxThreadsPerBlock
              << "\n";

    return 0;
}




