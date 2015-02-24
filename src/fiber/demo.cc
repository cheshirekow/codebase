/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of fiber.
 *
 *  fiber is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  fiber is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with fiber.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <fiber/fiber.h>
#include <iostream>

using namespace fiber;
typedef Matrix<double,3,3>   Mat3;
typedef Matrix<double,3,1>   Vec3;
typedef Quaternion<double>   Quat;

int main()
{
    std::cout << "here" << std::endl;
    Mat3 A;
    A << 0, 1, 2,
         3, 4, 5,
         6, 7, 8;


    std::cout << "A:" << std::endl;
    std::cout << A << std::endl;

    Mat3 B;
    for(int i=0; i < 3; i++)
        for(int j=0; j < 3; j++)
            B(i,j) = 10*(i*3+j);

    std::cout << "B:" << std::endl;
    std::cout << B << std::endl;

    std::cout << "A+B" << std::endl;
    std::cout << (A+B) << std::endl;

    std::cout << "B-A" << std::endl;
    std::cout << (B-A) << std::endl;

    std::cout << "A Transposed" << std::endl;
    std::cout << Transpose(A)   << std::endl;

    std::cout << "3*A:" << std::endl;
    std::cout << 3.0*A << std::endl;

    std::cout << "(1/2)*(B-A)^T:" << std::endl;
    std::cout << 0.5*Transpose(B-A) << std::endl;

    std::cout << "A*A:" << std::endl;
    std::cout << A*A << std::endl;

    std::cout << "A[(0,0),(1,1)]:" << std::endl;


    std::cout << "block(A,0,0,2,2) = view(B,0,0,2,2)" << std::endl;
    Block(A,0,0,2,2) = View(B,0,0,2,2);
    std::cout << A << std::endl;

    Vec3 c;
    c << 0, 1, 2;
    std::cout << "c:\n";
    std::cout << c << "\n";
    std::cout << "c':\n";
    std::cout << Transpose(c) << "\n";
    std::cout << "c'c:\n";
    std::cout << Transpose(c)*c << "\n";
    std::cout << "c dot c:\n";
    std::cout << Dot(c,c) << "\n\n";

    Quat q0(1,0,0,0);
    //Quat q1( normalize(Vec3(1,1,1)),M_PI/2);
    std::cout << "q0:\n" << q0 << "\n";
//    std::cout << "q1:\n" << q1 << "\n";
//    std::cout << "norm(q1):\n" << norm(q1) << "\n\n";
//    std::cout << "q0*q1:\n" << q0*q1 << "\n";
}











