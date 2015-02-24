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
 *  @file
 *
 *  @date   Nov 5, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_DUBINS_CURVES_FUNCS_H_
#define MPBLOCKS_DUBINS_CURVES_FUNCS_H_

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace mpblocks {
namespace   dubins {

/// wraps the input onto [-pi,pi]
template <typename Format_t>
__host__ __device__
Format_t clampRadian( Format_t a );

/// returns the counter clockwise (left) distance from a to b
template <typename Format_t>
__host__ __device__
Format_t ccwArc( Format_t a, Format_t b );

/// returns the counter clockwise (left) distance from a to b
template <typename Format_t>
__host__ __device__
Format_t leftArc( Format_t a, Format_t b );

/// returns the clockwise (right) distance from a to b
template <typename Format_t>
__host__ __device__
Format_t cwArc( Format_t a, Format_t b);

/// returns the clockwise (right) distance from a to b
template <typename Format_t>
__host__ __device__
Format_t rightArc( Format_t a, Format_t b);

} // dubins
} // mpblocks

#endif // MPBLOCKS_DUBINS_CURVES_FUNCS_H_
