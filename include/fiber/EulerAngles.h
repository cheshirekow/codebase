/*
 *  Copyright (C) 2015 Josh Bialkowski (jbialk@mit.edu)
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
 *  @date   Feb 23, 2014
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 */
#ifndef FIBER_EULER_ANGLES_H_
#define FIBER_EULER_ANGLES_H_

namespace fiber {
namespace euler {

/// A vector of euler angles.
/**
 *  The convention is that rotations are applied in the reverse order that they
 *  are specified in the vector. For instance the orientation described by an
 *  euler::Angles<Scalar,0,1,2>(phi, theta, psi) vector corresponds to an
 *  orientation described by a rotation of psi about the z-axis (2-axis),
 *  followed by a rotation of theta about the new y-axis (1-axis), followed by
 *  a rotation of phi about the new x-axis (0-axis).
 */
template <typename Scalar, int Axis1, int Axis2, int Axis3>
class Angles : public Matrix<Scalar, 3, 1> {
 public:
  typedef Matrix<Scalar, 3, 1> MatrixBase;

  Angles() {}

  /// We require at least two values so as to disambiguate the constructor
  /// from the construct-by-rvalue constructor
  Angles(Scalar a, Scalar b, Scalar c)
      : MatrixBase(a, b, c) {
  }

  Angles(const Quaternion<Scalar>& q) {
#if FIBER_USE_STATIC_ASSERT
    static_assert(
        _AnglesFromQuaternion<Scalar, Axis1, Axis2, Axis3>::HAS_SPECIALIZATION,
        "There is no conversion from quaternions to this euler angle set");
#endif
    _AnglesFromQuaternion<Scalar, Axis1, Axis2, Axis3>::Build(q, this);
  }

  friend void EulerAnglesToQuaternion<Scalar, Axis1, Axis2, Axis3>(
      const euler::Angles<Scalar, Axis1, Axis2, Axis3>& euler,
      Quaternion<Scalar>* q);
};

typedef Angles<double,_1,_2,_1> Angles121d;
typedef Angles<double,_1,_2,_3> Angles123d;
typedef Angles<double,_3,_1,_3> Angles313d;

typedef Angles<float,_1,_2,_1> Angles121f;
typedef Angles<float,_1,_2,_3> Angles123f;
typedef Angles<float,_3,_1,_3> Angles313f;

}  // namespace euler
}  // namespace fiber

#endif  // FIBER_EULER_ANGLES_H_
