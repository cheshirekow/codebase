/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of fiber.
 *
 *  fiber is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  fiber is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with fiber.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef MPBLOCKS_LINALG_H_
#define MPBLOCKS_LINALG_H_

namespace fiber {
/// simple template expression library for linear algebra
namespace linalg   {

} // namespace linalg
} // namespace fiber

#include <cassert>
#include <cmath>
#include <fiber/StreamAssignment.h>
#include <fiber/RValue.h>
#include <fiber/LValue.h>
#include <fiber/Transpose.h>
#include <fiber/Difference.h>
#include <fiber/Sum.h>
#include <fiber/Product.h>
#include <fiber/Scale.h>
#include <fiber/View.h>
#include <fiber/Matrix.h>
#include <fiber/Normalize.h>
#include <fiber/Identity.h>
#include <fiber/CrossMatrix.h>

#include <fiber/rotation_conversions.h>
#include <fiber/Quaternion.h>
#include <fiber/AxisAngle.h>
#include <fiber/EulerAngles.h>

#include <fiber/ostream.h>

#endif // MPBLOCKS_LINALG_H_
