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

#include <mpblocks/polynomial.h>
#include <iostream>
#include <cstdlib>

using namespace mpblocks::polynomial;
using namespace mpblocks::linalg;

typedef Polynomial<double,0>   Poly0;
typedef Polynomial<double,1>   Poly1;
typedef Polynomial<double,2>   Poly2;

typedef Polynomial<double,Dynamic>   Polynomial_t;
typedef Quotient< double, Polynomial_t, Polynomial_t >    Quotient_t;
typedef SturmSequence< double > Sturm_t;

typedef Matrix< Polynomial_t, 2,2 > PolyMat_t;
typedef Matrix< Polynomial_t, 2,1 > PolyVec_t;
typedef Matrix< double, 2, 1 >      Vec_t;


int main()
{
    std::vector<int> spec;

    Polynomial_t A(4);
    A << -42, 0, -12, 1;

    Polynomial_t B(2);
    B << -3, 1;

    Quotient_t quotient = A/B;

    Polynomial_t C(4);
    C << -42, 0, -12, 8;

    Polynomial_t D(6);
    D << 2, -10, -20, 0, 5, 1;

    preprint( spec, A );
    preprint( spec, B );
    preprint( spec, A+B );
    preprint( spec, B-A );
    preprint( spec, A*B );
    preprint( spec, quotient.q() );
    preprint( spec, quotient.r() );
    preprint( spec, C );
    preprint( spec, normalized(C) );
    preprint( spec, D );

    std::cout << "A:" << "\n";
    print( std::cout, A, spec ) << "\n";

    std::cout << "B:" << "\n";
    print( std::cout, B, spec ) << "\n";

    std::cout << "A+B" << "\n";
    print( std::cout, (A+B), spec ) << "\n";

    std::cout << "B-A" << "\n";
    print( std::cout, (B-A), spec ) << "\n";

    std::cout << "A*B" << "\n";
    print( std::cout, (A*B), spec ) << "\n";

    std::cout << "A/B" << "\n";
    print( std::cout << "    q: ", quotient.q(), spec ) << "\n";
    print( std::cout << "    r: ", quotient.r(), spec ) << "\n";
    print( std::cout << "q*B+r: ", ((quotient.q()*B)+quotient.r()), spec ) << "\n";

    print( std::cout << " C :\n    ", C, spec ) << "\n";
    print( std::cout << "|C|:\n    ", normalized(C), spec ) << "\n";

    std::cout << "\nsturm sequence of C:\n";
    Sturm_t sturmC(C);
    print( std::cout, sturmC ) << "\n";


    print( std::cout << " D :\n    ", D, spec ) << "\n";
    Sturm_t sturmD(D);
    std::cout << "\nsturm sequence of D:\n";
    print( std::cout, sturmD ) << "\n";

    std::cout << "sign changes: \n";
    std::cout
        << "   -10: " << sturmD.signChanges(-10) << "\n"
        << "    -5: " << sturmD.signChanges( -5) << "\n"
        << "    -4: " << sturmD.signChanges( -4) << "\n"
        << "    -3: " << sturmD.signChanges( -3) << "\n"
        << "    -2: " << sturmD.signChanges( -2) << "\n"
        << "    -1: " << sturmD.signChanges( -1) << "\n"
        << "     0: " << sturmD.signChanges(  0) << "\n"
        << "     1: " << sturmD.signChanges(  1) << "\n"
        << "     2: " << sturmD.signChanges(  2) << "\n"
        << "     5: " << sturmD.signChanges(  5) << "\n"
        << "    10: " << sturmD.signChanges( 10) << "\n"
        << "\n";

    PolyVec_t P0;
    P0[0].resize(1);
    P0[1].resize(1);

    P0[0] << std::cos(0.3*M_PI);
    P0[1] << std::sin(0.3*M_PI);

    std::cout << "P0:\n" << P0 << "\n";

    PolyMat_t P1;
    P1(0,0) << 1,0,9;
    P1(0,1) << 0,-6;
    P1(1,0) << 0,6;
    P1(1,1) << 1,0,9;

    std::cout << "P1:\n" << P1 << "\n";
    std::cout << "P1 * P0:\n" << defer(P0 * P1) << "\n";
    std::cout << "P0 dot P0: \n" << transpose(P0)*P0 << "\n";
    std::cout << "P0 dot P0: \n" << dot(P0,P0) << "\n";
    std::cout << std::endl;

    Vec_t v;
    v << 3, 3;

//    Matrix<Poly2,2,2> P2;
//    for(int i=0; i < 2; i++)
//        for(int j=0; j<2; j++)
//            P2(i,j) = Poly2( rand() % 10, rand() % 10, rand() % 10 );
//
//    Matrix<Poly0,2,2> P3;
//    for(int i=0; i < 2; i++)
//        for(int j=0; j<2; j++)
//            P3(i,j) = Poly0( rand() % 10 );
}











