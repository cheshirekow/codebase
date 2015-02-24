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
#include <mpblocks/cuda/polynomial/IntList.h>

using namespace mpblocks::cuda::polynomial;

const char* bool_str( bool val )
{
    return val ? "true" : "false";
}

int main()
{
    typedef intlist::construct3<0,2,4>::result List1;
    typedef intlist::construct3<0,1,5>::result List2;
    typedef intlist::construct<0,1,3,8,6,2>::result List3;
    typedef intlist::construct<0,1,2,2,1,2,1,0,0,2>::result List4;

    std::cout << "List1 : " << Printer< List1 >() << "\n";
    std::cout << "length: " << intlist::size< List1 >::value << "\n";
    std::cout << "contains 0: "
              << bool_str( intlist::contains<List1,0>::value ) << "\n";
    std::cout << "contains 1: "
              << bool_str( intlist::contains<List1,1>::value ) << "\n";
    std::cout << "contains 2: "
              << bool_str( intlist::contains<List1,2>::value ) << "\n";
    std::cout << "contains 3: "
              << bool_str( intlist::contains<List1,3>::value ) << "\n";
    std::cout << "contains 4: "
              << bool_str( intlist::contains<List1,4>::value ) << "\n";

    std::cout << "List2 : " << Printer< List2 >() << "\n";
    std::cout << "List1 + List2 : "
      << Printer< intlist::join<List1,List2>::result >() << "\n";
    std::cout << "List1 U List2 : "
      << Printer< intlist::make_union<List1,List2>::result >() << "\n";


    std::cout << "List3 : " << Printer<List3>() << "\n";
    std::cout << "length: " << intlist::size< List3 >::value << "\n";
    std::cout << "List3[2,2] "
        << Printer< intlist::sublist<List3,2,2>::result >() << "\n";
    std::cout << "List3[3,4] "
        << Printer< intlist::sublist<List3,3,4>::result >() << "\n";
    std::cout << "List3 head(3) "
        << Printer< intlist::head<List3,3>::result >() << "\n";
    std::cout << "List3 tail(2) "
        << Printer< intlist::tail<List3,2>::result >() << "\n";
    std::cout << "List3 sorted "
        << Printer< intlist::merge_sort<List3>::result >() << "\n";


    std::cout << "List4   : " << Printer< List4 >() << "\n";
    typedef intlist::merge_sort< List4 >::result List4sorted;
    std::cout << "sorted  : " << Printer< List4sorted > () << "\n";
    typedef intlist::strip_dups< List4sorted >::result List4stripped;
    std::cout << "stripped: " << Printer< List4stripped >() << "\n";
    std::cout << "max     : " << intlist::max< List4 >::value << "\n";

    std::cout << "all pairs sum List1+List2:"
        << Printer< intlist::allpairs_sum<List1,List2>::result >() << "\n";
}











