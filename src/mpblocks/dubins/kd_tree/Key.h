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
 *  @file   mpblocks/kd_tree/r2_s1/Key.h
 *
 *  @date   Nov 21, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_KD_TREE_KEY_H_
#define MPBLOCKS_DUBINS_KD_TREE_KEY_H_



namespace mpblocks {
namespace   dubins {
namespace  kd_tree {


/// key for priority queue
template <class Format>
struct Key
{
    typedef typename Traits<Format>::Node   Node_t;
    typedef Key<Format>                     Key_t;

    Format   d2; //< distance
    int      id; //< solver id
    Node_t*  n;  //< nearest node

    struct LessThan
    {
        bool operator()( const Key_t& a, const Key_t& b ) const
        {
            if( a.d2 < b.d2 )
                return true;
            if( a.d2 > b.d2 )
                return false;
            return (a.n < b.n );
        }
    };
};


} // namespace kd_tree
} // namespace dubins
} // namespace mpblocks



#endif // SEARCHKEY_H_
