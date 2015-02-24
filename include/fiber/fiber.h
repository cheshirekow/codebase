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
#include <fiber/stream_assignment.h>
#include <fiber/rvalue.h>
#include <fiber/lvalue.h>
#include <fiber/transpose.h>
#include <fiber/difference.h>
#include <fiber/sum.h>
#include <fiber/product.h>
#include <fiber/scale.h>
#include <fiber/view.h>
#include <fiber/matrix.h>
#include <fiber/normalize.h>
#include <fiber/identity.h>
#include <fiber/cross_matrix.h>

#include <fiber/rotation_conversions.h>
#include <fiber/quaternion.h>
#include <fiber/axis_angle.h>
#include <fiber/euler_angles.h>

#include <fiber/ostream.h>

#endif // MPBLOCKS_LINALG_H_
