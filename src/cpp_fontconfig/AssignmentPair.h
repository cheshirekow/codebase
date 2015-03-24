/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of cppfontconfig.
 *
 *  cppfontconfig is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  cppfontconfig is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with cppfontconfig.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   include/cppfontconfig/AssignmentPair.h
 *
 *  @date   Feb 5, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef CPPFONTCONFIG_ASSIGNMENTPAIR_H_
#define CPPFONTCONFIG_ASSIGNMENTPAIR_H_


namespace fontconfig {

/// allows an error to be returned with a result in a single expression
template <typename T1, typename T2 >
struct RValuePair
{
    T1  p1;
    T2  p2;

    RValuePair( T1 p1_in, const T2 p2_in):
        p1(p1_in),
        p2(p2_in)
    {}
};

/// allows an error to be returned with a result in a single expression
template <typename T1, typename T2 >
struct LValuePair
{
    T1&  p1;
    T2&  p2;

    LValuePair( T1& p1_ref, T2& p2_ref):
        p1(p1_ref),
        p2(p2_ref)
    {}

    void operator=( const RValuePair<T1,T2>& copy )
    {
        p1 = copy.p1;
        p2 = copy.p2;
    }
};

} // namespace fontconfig















#endif // ERRORPAIR_H_
