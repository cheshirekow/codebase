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

#ifndef MPBLOCKS_POLYNOMIAL_OSTREAM_H_
#define MPBLOCKS_POLYNOMIAL_OSTREAM_H_

#include <iostream>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <vector>

namespace   mpblocks {
namespace polynomial {



template <typename Scalar, class Exp>
std::ostream& print( std::ostream& out,
                    RValue<Scalar,Exp> const& p,
                    const std::string& var = "x")
{
    for(int i=0; i < p.size(); i++)
    {
        out << p[i]
            << " " << var << "^"
            << i;
        if( i < p.size()-1 )
            out << " + ";
    }

    return out;
}

template <typename Scalar, class Exp>
std::ostream& print( std::ostream& out,
                    RValue<Scalar,Exp> const& p,
                    const std::vector<int>& spec,
                    const std::string& var = "x")
{
    std::stringstream strm;
    for(int i=0; i < p.size(); i++)
    {
        strm.str("");
        strm << p[i]
            << " " << var << "^"
            << i;
        out << std::setw(spec[i])
            << strm.str();
        if( i < p.size()-1 )
            out << " + ";
    }

    return out;
}


template <typename Scalar, class Exp>
void preprint( std::vector<int>& spec,
                    RValue<Scalar,Exp> const& p,
                    const std::string& var = "x")
{
    std::stringstream out;
    if( spec.size() < p.size() )
        spec.resize( p.size(), 0 );

    for(int i=0; i < p.size(); i++)
    {
        out.str("");
        out << p[i]
            << " " << var << "^"
            << i;
        if( out.str().size() > spec[i] )
            spec[i] = out.str().size();
    }
}


template <typename Scalar, class Exp>
std::ostream& operator<<( std::ostream& out,
                        RValue<Scalar,Exp> const& p )
{
    return print(out,p);
}

template <typename Scalar>
std::ostream& print( std::ostream& out,
                    SturmSequence<Scalar> const& seq,
                    const std::string& var="x")
{
    std::vector<int> spec( seq[0].size(), 0 );

    for(int i=0; i < seq.size(); i++)
        preprint(spec,seq[i],var);
    for(int i=0; i < seq.size(); i++)
        print(out,seq[i],spec,var) << "\n";
    return out;
}

template <typename Scalar>
std::ostream& operator<<( std::ostream& out,
                        SturmSequence<Scalar> const& seq )
{
    return print(out,seq);
}






} // polynomial
} // mpblocks









#endif // IOSTREAM_H_
