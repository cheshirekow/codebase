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
#ifndef MPBLOCKS_CLARKSON93_BITMEMBER_H_
#define MPBLOCKS_CLARKSON93_BITMEMBER_H_

#include <bitset>

namespace   mpblocks {
namespace clarkson93 {

/// dummy class which allows us to use SNFINAE
struct BitMemberBase{};

/// indicates membership into a number of sets by a bitfield
template <typename Enum, unsigned int Size>
struct BitMember:
    public BitMemberBase,
    public std::bitset<Size>
{
    void addTo( Enum setBit )
    {
        (*this)[setBit] = true;
    }

    void removeFrom( Enum setBit )
    {
        (*this)[setBit] = false;
    }

    bool isMemberOf( Enum setBit )
    {
        return (*this)[setBit];
    }
};


} // namespace clarkson93
} // namespace mpblocks





#endif // BITMEMBER_H_
