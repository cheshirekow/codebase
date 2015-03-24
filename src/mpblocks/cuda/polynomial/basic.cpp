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
#include <mpblocks/cuda/polynomial/Polynomial.h>
#include <mpblocks/cuda/polynomial/ostream.h>
#include <mpblocks/cuda/polynomial/Sum.h>
#include <mpblocks/cuda/polynomial/Difference.h>
#include <mpblocks/cuda/polynomial/Construct.h>
#include <mpblocks/cuda/polynomial/StreamAssignment.h>
#include <mpblocks/cuda/polynomial/polyval.h>
#include <mpblocks/cuda/polynomial/Normalized.h>
#include <mpblocks/cuda/polynomial/Negative.h>
#include <mpblocks/cuda/polynomial/differentiate.h>
#include <mpblocks/cuda/polynomial/Product.h>
#include <mpblocks/cuda/polynomial/Quotient.h>
#include <mpblocks/cuda/polynomial/SturmSequence.h>


using namespace mpblocks::cuda::polynomial;

int main()
{
    typedef intlist::construct<0,1,3>::result SpecA;
    typedef intlist::construct<2,4>::result   SpecB;

    Polynomial<float, SpecA > pA(0,1,3);
    Polynomial<float, SpecB > pB(2,4);

    std::cout << "pA: " << pA << "\n";
    std::cout << "pB: " << pB << "\n";
    std::cout << "setting pA by set<i>\n";
    set<0>(pA) = 1;
    set<1>(pA) = 2;
    set<3>(pA) = 4;
    std::cout << "pA: " << pA << "\n";
    std::cout << "retrieving pA by get<i>\n";
    std::cout << "coefficients: "
                 "\n   0: " << get<0>(pA)
              << "\n   1: " << get<1>(pA)
              << "\n   2: " << get<2>(pA)
              << "\n   3: " << get<3>(pA)
              << "\n   4: " << get<4>(pA)
              << "\n";

    std::cout << "     pA + pB : " << ( pA + pB ) << "\n";
    std::cout << "     pA - pB : " << ( pA - pB ) << "\n";

    typedef typename product_detail::product_spec<SpecA,SpecB>::result SpecAB;
    std::cout << "Product spec   " << product_detail::SpecPrinter< SpecAB >() << "\n";
    std::cout << "     pA * pB : " << ( pA * pB ) << "\n";

    Polynomial<float, intlist::construct<0,1,2,3,4>::result > pC =
            pA + pB;
    std::cout << "pC = pA + pB : " << pC << "\n";

    std::cout << "setting pD by writing out the math:\n";
    using namespace param_key;
    using namespace coefficient_key;
    Polynomial<float, intlist::construct<0,2>::result > pD =
            1.2f * (s^_0) + 1.35f * (s^_2);
    std::cout << "pD: " << pD << "\n";

    std::cout << "setting pE by stream assignment:\n";
    Polynomial<float, intlist::construct<0,1,3,5>::result > pE;
    pE << 0.23, 1.25, 2.3, 0.12;
    std::cout << "pE            : " << pE << "\n";
    std::cout << "pE normalized : " << normalized(pE) << "\n";
    std::cout << "-pE           : " << -pE << "\n";

    std::cout << "polyval pE(2.5): " << polyval( pE, 2.5 ) << "\n";

    typedef intlist::construct<3,5,8>::result pF_spec;
    typedef DerivativeSpec<pF_spec,1>::result d1_pF_spec;
    typedef DerivativeSpec<pF_spec,2>::result d2_pF_spec;
    typedef DerivativeSpec<pF_spec,3>::result d3_pF_spec;
    typedef DerivativeSpec<pF_spec,4>::result d4_pF_spec;

    Polynomial<float, pF_spec > pF(1,1,1);
    Polynomial<float, d1_pF_spec > d1pF = d_ds<1>(pF);
    Polynomial<float, d2_pF_spec > d2pF = d_ds<2>(pF);
    Polynomial<float, d3_pF_spec > d3pF = d_ds<3>(pF);
    Polynomial<float, d4_pF_spec > d4pF = d_ds<4>(pF);

    std::cout << "pF and it's derivitives: \n"
            << "\n 0: " << pF
            << "\n 1: " << d1pF
            << "\n 2: " << d2pF
            << "\n 3: " << d3pF
            << "\n 4: " << d4pF
            << "\n";

    typedef intlist::construct<0,1,2>::result   SpecNum;
    typedef intlist::construct<0,1>::result     SpecDen;
    typedef Polynomial<float, SpecNum >         ExpNum;
    typedef Polynomial<float, SpecDen >         ExpDen;
    typedef Quotient<float,ExpNum,SpecNum,ExpDen,SpecDen>   Quotient_t;

    ExpNum Num(-10,-9,1);
    ExpDen Den(1,1);
    Quotient_t result = Num / Den;

    std::cout << "\n            : " << Num
              << "\n divided by : " << Den
              << "\n------------------------------------------------"
              << "\n   quotient : " << result.q
              << "\n  remainder : " << result.r
              << "\n";

    std::cout << "done\n";

    typedef intlist::construct<0,1,2,3,4,5>::result SpecG;
    Polynomial<float, SpecG > pG( 2, -10, -20, 0, 5, 1 );
    SturmSequence< float,5 > sG(pG);

    std::cout << "pG : "   << pG << "\n";
    std::cout << "sturm: "
              << "\n p0: " << get<0>(sG)
              << "\n p1: " << get<1>(sG)
              << "\n p2: " << get<2>(sG)
              << "\n p3: " << get<3>(sG)
              << "\n p4: " << get<4>(sG)
              << "\n p4: " << get<5>(sG)
              << "\n";
    std::cout << "sign changes: \n";
    std::cout
        << "   -10: " << sG.signChanges(-10) << "\n"
        << "    -5: " << sG.signChanges( -5) << "\n"
        << "    -4: " << sG.signChanges( -4) << "\n"
        << "    -3: " << sG.signChanges( -3) << "\n"
        << "    -2: " << sG.signChanges( -2) << "\n"
        << "    -1: " << sG.signChanges( -1) << "\n"
        << "     0: " << sG.signChanges(  0) << "\n"
        << "     1: " << sG.signChanges(  1) << "\n"
        << "     2: " << sG.signChanges(  2) << "\n"
        << "     5: " << sG.signChanges(  5) << "\n"
        << "    10: " << sG.signChanges( 10) << "\n"
        << "\n";
}











