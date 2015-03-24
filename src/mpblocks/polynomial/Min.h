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
 *  @file   include/mpblocks/polynomial/Min.h
 *
 *  @date   Jan 16, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_POLYNOMIAL_MIN_H_
#define MPBLOCKS_POLYNOMIAL_MIN_H_


namespace   mpblocks {
namespace polynomial {

template <int Val1, int Val2, bool First=true>
struct MinSelect
{
    enum
    {
        VALUE=Val1
    };
};

template <int Val1, int Val2>
struct MinSelect< Val1, Val2, false >
{
    enum
    {
        VALUE=Val2
    };
};

template <int Val1, int Val2>
struct Min
{
    enum
    {
        VALUE= MinSelect<Val1,Val2,(Val1 < Val2)>::VALUE
    };
};




} // polynomial
} // mpblocks





#endif // MINVALUE_H_
