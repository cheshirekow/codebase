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
 *  \file   mpblocks/dubins/curves.h
 *
 *  \date   Oct 30, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_H_
#define MPBLOCKS_DUBINS_CURVES_H_

#include <Eigen/Dense>

#include <mpblocks/dubins/curves/types.h>
#include <mpblocks/dubins/curves/funcs.h>
#include <mpblocks/dubins/curves/path.h>
#include <mpblocks/dubins/curves/result.h>

namespace mpblocks     {
namespace dubins       {
/// classes for solving dubins curves, optimal primitives for shortest path
/// between two states of a dubins vehicle
namespace curves_eigen {

} // curves
} // dubins
} // mpblocks

#include <mpblocks/dubins/curves_eigen/funcs.h>
#include <mpblocks/dubins/curves_eigen/integrator.h>
#include <mpblocks/dubins/curves_eigen/query.h>
#include <mpblocks/dubins/curves_eigen/opposite.h>
#include <mpblocks/dubins/curves_eigen/solver.h>
#include <mpblocks/dubins/curves_eigen/trig_wrap.h>

#include <mpblocks/dubins/curves_eigen/hyper_rect.h>

#endif // MPBLOCKS_DUBINS_CURVES_H_
