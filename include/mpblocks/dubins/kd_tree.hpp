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
 *  @file   /home/josh/Codes/cpp/mpblocks2/dubins/include/mpblocks/dubins/kd_tree.hpp
 *
 *  @date   Jul 6, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_KD_TREE_HPP_
#define MPBLOCKS_DUBINS_KD_TREE_HPP_

#include <mpblocks/dubins/kd_tree.h>
#include <mpblocks/kd_tree.hpp>


namespace mpblocks {
namespace   dubins {
namespace  kd_tree {


template < class Scalar >
Traits<Scalar>::HyperRect::HyperRect()
{
    minExt.fill(0);
    maxExt.fill(0);
}



template < class Scalar >
Scalar Traits<Scalar>::HyperRect::measure()
{
    Scalar s = 1.0;
    for(unsigned int i=0; i < minExt.rows(); i++)
        s *= maxExt[i] - minExt[i];

    return s;
}




} // namespace kd_tree
} // namespace dubins
} // namespace mpblocks



#include <mpblocks/dubins/kd_tree/Distance.hpp>
#include <mpblocks/dubins/kd_tree/KNearest.hpp>
#include <mpblocks/dubins/kd_tree/Nearest.hpp>

#endif // KD_TREE_HPP_
