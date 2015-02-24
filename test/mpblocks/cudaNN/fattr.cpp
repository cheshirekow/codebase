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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_nn/test/simple/main.cpp
 *
 *  @date   Oct 7, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */


#include <vector>
#include <map>
#include <iostream>
#include <cstdio>
#include <mpblocks/cuda.h>
#include <mpblocks/cudaNN/fattr.h>

using namespace mpblocks;

int main(int argc, char** argv)
{
    cuda::setDevice(0);
    cudaNN::fattrMap_t fattrMap;
    cudaNN::get_fattr<float,4>( fattrMap );

    for( cudaNN::fattrMap_t::iterator iter = fattrMap.begin();
            iter != fattrMap.end(); ++iter )
    {
        std::cout << "Kernel: " << iter->first
                  << "\nptx version: " << iter->second.ptxVersion
                  << "\nbin version: " << iter->second.binaryVersion
                  << "\n  registers: " << iter->second.numRegs
                  << "\n const size: " << iter->second.constSizeBytes
                  << "\n local size: " << iter->second.localSizeBytes
                  << "\nshared size: " << iter->second.sharedSizeBytes
                  << "\nmax / block: " << iter->second.maxThreadsPerBlock
                  << "\n\n";
    }



    return 0;
}







