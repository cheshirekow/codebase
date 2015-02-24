/*
 *  Copyright (C) 2014 Josh Bialkowski (josh.bialkowski@gmail.com)
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
 *  @date   Nov 12, 2014
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief  compatability macros between host code and device code
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA_PORTABLE_H_
#define MPBLOCKS_DUBINS_CURVES_CUDA_PORTABLE_H_

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#endif  // MPBLOCKS_DUBINS_CURVES_CUDA_PORTABLE_H_
