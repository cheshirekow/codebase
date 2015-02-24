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
 *  @file   include/mpblocks/cuda/linalg/iostream.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG_IOSTREAM_H_
#define MPBLOCKS_CUDA_LINALG_IOSTREAM_H_

#include <iostream>
#include <cstdio>
#include <iomanip>

namespace mpblocks {
namespace cuda     {
namespace linalg   {

template <typename Scalar, class Mat>
std::ostream& operator<<( std::ostream& out, RValue<Scalar,Mat> const& M)
{
    for(int i=0; i < M.rows(); i++)
    {
        for(int j=0; j < M.cols(); j++)
        {
            out << std::setw(8)
                    << std::setiosflags(std::ios::fixed)
                    << std::setprecision(4) << M(i,j);
        }
        out << "\n";
    }

    return out;
}






} // linalg
} // cuda
} // mpblocks









#endif // IOSTREAM_H_
