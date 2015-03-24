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
 *  @file   test/lingalg/matrixstream.cpp
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <iostream>
#include <mpblocks/cuda/fakecuda.h>
#include <mpblocks/cuda/linalg2.h>


using namespace mpblocks::cuda::linalg2;
typedef Matrix<double,3,3>   Matrix_t;


int main()
{
    Matrix_t A;
    A << 0, 1, 2,
         3, 4, 5,
         6, 7, 8;

    Matrix_t C[2];
    C[0] = A;
    C[1] = A;
//
//
    std::cout << "A:" << std::endl;
    std::cout << A << std::endl;

    Matrix_t B = 3.0*A;

    std::cout << "B:" << std::endl;
    std::cout << B << std::endl;

    std::cout << "A+B" << std::endl;
    std::cout << (A+B) << std::endl;

    std::cout << "B-A" << std::endl;
    std::cout << (B-A) << std::endl;

    std::cout << "A transposed" << std::endl;
    std::cout << transpose(A)   << std::endl;

    std::cout << "3*A:" << std::endl;
    std::cout << 3.0*A << std::endl;

    std::cout << "(1/2)*(B-A)^T:" << std::endl;
    std::cout << 0.5*transpose(B-A) << std::endl;

    std::cout << "A*A:" << std::endl;
    std::cout << A*A << std::endl;

    std::cout << "B = A*A:" << std::endl;
    B = C[0]*C[1];
    std::cout << B << std::endl;

    std::cout << "Setting A(0,0) to 5\n";
    set<0,0>(A) = 5;
    std::cout << A << std::endl;

    std::cout << "Upper right block of A:\n";
    std::cout << view<0,1,2,2>(A) << std::endl;

    std::cout << "Upper right element of upper block of A:\n";
    std::cout << get<0,1>( view<0,1,2,2>(A) ) << std::endl;

    std::cout << "A vector V:\n";
    Matrix<double,4,1> V;
    V << 1, 2, 3, 4;
    std::cout << transpose(V) << std::endl;

    std::cout << "The norm of V:\n";
    std::cout << "squared: " << norm_squared(V) << "\n";
    std::cout << "sqrt: "    << norm(V)         << "\n\n";

    std::cout << "V normalized:\n"
              << normalize(V) << "\n";

    std::cout << "last two elements of V:\n"
              << view<2,2>(V) << "\n";

    std::cout << "rotation by M_PI/2:\n"
              << Rotation2d<double>(M_PI/2) << "\n";
//
//    std::cout << "A[(0,0),(1,1)]:" << std::endl;
//    std::cout << view(A,0,0,2,2) << std::endl;
//
//    std::cout << "block(A,0,0,2,2) = view(B,0,0,2,2)" << std::endl;
//    block(A,0,0,2,2) = view(B,0,0,2,2);
//    std::cout << A << std::endl;
}











