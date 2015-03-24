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
 *  @file   /home/josh/Codes/cpp/mpblocks2/dubins/include/mpblocks/dubins/kdtree.h
 *
 *  @date   Jul 5, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_KDTREE_H_
#define MPBLOCKS_DUBINS_KDTREE_H_

#include <mpblocks/kd_tree.h>

namespace mpblocks {
namespace   dubins {
namespace  kd_tree {

template <typename Format>
struct Traits
{
    typedef Format Format_t;
    static const unsigned NDim = 3;

    struct HyperRect:
        public curves_eigen::hyper::HyperRect<Format>
    {
        typedef curves_eigen::hyper::HyperRect<Format> Base_t;

        using Base_t::minExt;
        using Base_t::maxExt;

        HyperRect();

        /// return the measure of the hypercube
        Format_t measure();
    };

    struct Node:
        public mpblocks::kd_tree::Node<Traits>
    {
        unsigned int idx;
    };

};


} // namespace kd_tree
} // namespace dubins
} // namespace mpblocks



#include <mpblocks/dubins/kd_tree/Distance.h>
#include <mpblocks/dubins/kd_tree/Key.h>
#include <mpblocks/dubins/kd_tree/KNearest.h>
#include <mpblocks/dubins/kd_tree/Nearest.h>














#endif // KDTREE_H_
