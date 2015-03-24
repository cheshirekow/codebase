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
 *  @file   include/mpblocks/polynomial/ostream.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_OSTREAM_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_OSTREAM_H_

#include <iostream>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <vector>

#include <mpblocks/cuda/polynomial/Polynomial.h>

namespace   mpblocks {
namespace       cuda {
namespace polynomial {



template < typename Scalar, class Exp, class Spec >
struct PolyPrinter{};

template < class Scalar, class Exp, int Head, class Tail >
struct PolyPrinter< Scalar, Exp, IntList<Head,Tail> >
{
    static void print( std::ostream& out, const Exp& exp)
    {
        Scalar val = get<Head>(exp);
        if( val != 0 )
            out << val << " s^" << Head << " + ";
        PolyPrinter< Scalar,Exp,Tail>::print(out,exp);
    }
};

template < class Scalar, class Exp, class Tail >
struct PolyPrinter< Scalar, Exp, IntList<1,Tail> >
{
    static void print( std::ostream& out, const Exp& exp)
    {
        Scalar val = get<1>(exp);
        if( val != 0 )
            out << val << " s + ";
        PolyPrinter< Scalar,Exp,Tail>::print(out,exp);
    }
};

template < class Scalar, class Exp, class Tail >
struct PolyPrinter< Scalar, Exp, IntList<0,Tail> >
{
    static void print( std::ostream& out, const Exp& exp)
    {
        Scalar val = get<0>(exp);
        if( val != 0 )
            out << val << " + ";
        PolyPrinter< Scalar,Exp,Tail>::print(out,exp);
    }
};


template < class Scalar, class Exp>
struct PolyPrinter< Scalar, Exp, intlist::Terminal >
{
    static void print( std::ostream& out, const Exp& exp )
    {}
};


template < class Scalar, class Exp, class Spec>
std::ostream& operator<<( std::ostream& out,
                            const RValue<Scalar,Exp,Spec>& exp )
{
    PolyPrinter<Scalar,Exp,Spec>::print(out, static_cast<const Exp&>(exp) );
    return out;
}




} // polynomial
} // cuda
} // mpblocks









#endif // IOSTREAM_H_
