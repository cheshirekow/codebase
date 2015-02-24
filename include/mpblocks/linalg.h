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
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef MPBLOCKS_LINALG_H_
#define MPBLOCKS_LINALG_H_

namespace mpblocks {
/// simple template expression library for linear algebra
namespace linalg   {

} // namespace linalg
} // namespace mpblocks

#include <cassert>
#include <cmath>
#include <mpblocks/linalg/StreamAssignment.h>
#include <mpblocks/linalg/RValue.h>
#include <mpblocks/linalg/LValue.h>
#include <mpblocks/linalg/Transpose.h>
#include <mpblocks/linalg/Difference.h>
#include <mpblocks/linalg/Sum.h>
#include <mpblocks/linalg/Product.h>
#include <mpblocks/linalg/Scale.h>
#include <mpblocks/linalg/View.h>
#include <mpblocks/linalg/Matrix.h>
#include <mpblocks/linalg/Normalize.h>
#include <mpblocks/linalg/Identity.h>
#include <mpblocks/linalg/CrossMatrix.h>

#include <mpblocks/linalg/rotation_conversions.h>
#include <mpblocks/linalg/Quaternion.h>
#include <mpblocks/linalg/AxisAngle.h>
#include <mpblocks/linalg/EulerAngles.h>

#include <mpblocks/linalg/ostream.h>

#endif // MPBLOCKS_LINALG_H_
