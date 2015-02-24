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
 *  @file   /home/josh/Codes/cpp/mpblocks2/kdtree/include/mpblocks/kd2/BinaryKey.h
 *
 *  @date   Nov 3, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_ROBOT_NN_BINARYKEY_H_
#define MPBLOCKS_ROBOT_NN_BINARYKEY_H_





namespace mpblocks {
namespace robot_nn {


enum BinaryKey
{
    MIN  = 0,
    MAX  = 1
};

inline BinaryKey other( const BinaryKey& key )
{
    return BinaryKey( 1-key );
}



} //< namespace kd2
} //< namespace mpblocks




#endif // BINARYKEY_H_
