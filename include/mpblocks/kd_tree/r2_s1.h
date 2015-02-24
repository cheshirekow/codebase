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
 *  @file   mpblocks/kd_tree/r2_s1.h
 *
 *  @date   Nov 21, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_KD_TREE_R2_S1_H_
#define MPBLOCKS_KD_TREE_R2_S1_H_


namespace  mpblocks {
namespace   kd_tree {

/// search implementations for the manifold @f \mathbb{R}^2 \times S^1 @f,
/// representing rigid bodies in 2d under translation and rotation
namespace r2_s1 {

template <class Traits>     struct Distance;
template <class Traits>     struct Key;
template <class Traits,
          template <class> class Allocator>
                            class  KNearest;
template <class Traits>     class  Nearest;



} // namespace r2_s1
} // namespace kd_tree
} // namespace mpblocks



#include <mpblocks/kd_tree/r2_s1/Distance.h>
#include <mpblocks/kd_tree/r2_s1/Key.h>
#include <mpblocks/kd_tree/r2_s1/HyperRect.h>
#include <mpblocks/kd_tree/r2_s1/KNearest.h>
#include <mpblocks/kd_tree/r2_s1/Nearest.h>






#endif // R2_S1_H_
