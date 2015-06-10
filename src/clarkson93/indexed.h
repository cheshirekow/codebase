/*
 *  Copyright (C) 2013 Josh Bialkowski (jbialk@mit.edu)
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
 *  @file   clarkson93/Indexed.h
 *
 *  @date   Jun 20, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CLARKSON93_INDEXED_H_
#define MPBLOCKS_CLARKSON93_INDEXED_H_



namespace   mpblocks {
namespace clarkson93 {


/// priority queue node
template <typename Index_t, typename Value_t>
struct Indexed
{
    typedef Indexed<Index_t,Value_t> This;

    Index_t   idx;
    Value_t   val;

    Indexed(){}
    Indexed( Index_t idx, Value_t val ): idx(idx), val(val){}

    static bool lessThan( const This& a, const This& b )
    {
        if( a.idx < b.idx )
            return true;
        else if( a.idx == b.idx )
        {
            if( a.val < b.val )
                return true;
            else
                return false;
        }
        else
            return false;
    }

    static bool greaterThan( const This& a, const This& b )
    {
        if( a.idx > b.idx )
            return true;
        else if( a.idx == b.idx )
        {
            if( a.val > b.val )
                return true;
            else
                return false;
        }
        else
            return false;
    }

    struct Less
    {
        bool operator()( const This& a, const This& b )
        {
            return This::lessThan(a,b);
        }
    };

    struct Greater
    {
        bool operator()( const This& a, const This& b )
        {
            return This::greaterThan(a,b);
        }
    };
};

template <typename Index_t, typename Value_t>
bool operator<( const Indexed<Index_t,Value_t>& a,
                const Indexed<Index_t,Value_t>& b )
{
    return Indexed<Index_t,Value_t>::lessThan(a,b);
}

template <typename Index_t, typename Value_t>
bool operator>( const Indexed<Index_t,Value_t>& a,
                const Indexed<Index_t,Value_t>& b )
{
    return Indexed<Index_t,Value_t>::greaterThan(a,b);
}


} // namespace clarkson93
} // namespace mpblocks





#endif // INDEXED_H_
