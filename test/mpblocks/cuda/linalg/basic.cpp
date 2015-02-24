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

#include <mpblocks/cuda/fakecuda.h>
#include <mpblocks/cuda/linalg.h>
#include <iostream>

using namespace mpblocks::cuda::linalg;
typedef Matrix<double,3,3>   Matrix_t;


int main()
{
    std::cout << "here" << std::endl;
    Matrix_t A;
    A << 0, 1, 2,
         3, 4, 5,
         6, 7, 8;


    std::cout << "A:" << std::endl;
    std::cout << A << std::endl;

    Matrix_t B;
    for(int i=0; i < 3; i++)
        for(int j=0; j < 3; j++)
            B(i,j) = 10*(i*3+j);

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

    std::cout << "A[(0,0),(1,1)]:" << std::endl;
    std::cout << view(A,0,0,2,2) << std::endl;

    std::cout << "block(A,0,0,2,2) = view(B,0,0,2,2)" << std::endl;
    block(A,0,0,2,2) = view(B,0,0,2,2);
    std::cout << A << std::endl;

    Matrix<double,2,2> C;
    C << 1, 0,
         0, 1;
    Rotation2d<double> R(M_PI/4.0);
    std::cout << "C:" << std::endl;
    std::cout << C << std::endl;
    std:: cout << "R:" << std::endl;
    std::cout << R << std::endl;

    std::cout << "R*C:" << std::endl;
    std::cout << R*C << std::endl;

}











