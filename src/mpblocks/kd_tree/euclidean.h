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
 *  @file   mpblocks/kd_tree/euclidean.h
 *
 *  @date   Nov 21, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_H_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_H_


namespace  mpblocks {
namespace   kd_tree {

/// search implementations for a euclean metric space, distance is euclidean
/// distance, ball is a euclidean ball
namespace euclidean {

template <class Traits,
          template <class> class Allocator>
                            class  Ball;
template <class Traits>     struct Distance;
template <class Traits>     struct Key;
template <class Traits>     struct HyperRect;

template <class Traits,
          template <class> class Allocator>
                            class  KNearest;
template <class Traits,
          template <class> class Allocator>
                            class  KNearestBall;
template <class Traits,
          template <class> class Allocator>
                            class  KNearestBallCenter;
template <class Traits>     class  Nearest;



} // namespace euclidean
} // namespace kd_tree
} // namespace mpblocks



#include <mpblocks/kd_tree/euclidean/Ball.h>
#include <mpblocks/kd_tree/euclidean/Distance.h>
#include <mpblocks/kd_tree/euclidean/Key.h>
#include <mpblocks/kd_tree/euclidean/HyperRect.h>
#include <mpblocks/kd_tree/euclidean/KNearest.h>
#include <mpblocks/kd_tree/euclidean/KNearestBall.h>
#include <mpblocks/kd_tree/euclidean/KNearestBallCenter.h>
#include <mpblocks/kd_tree/euclidean/Nearest.h>






#endif // EUCLIDEAN_H_
