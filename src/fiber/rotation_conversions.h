/*
 *  Copyright (C) 2014 Josh Bialkowski (jbialk@mit.edu)
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
 *  @date   Dec 10, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */
#ifndef FIBER_ROTATION_CONVERSIONS_H_
#define FIBER_ROTATION_CONVERSIONS_H_


namespace fiber {

// forward declare the types
template <typename Scalar>
class AxisAngle;
template <typename Scalar, int Axis>
class CoordinateAxisAngle;
template <typename Scalar>
class Quaternion;
namespace euler {
template <typename Scalar, int Axis1, int Axis2, int Axis3>
class Angles;
}  // namespace euler


/**
 *  We follow the counter-clockwise right hand convention. For example, if r is
 *  a vector in the x-y plane and the axis of rotation is the z-axis, then
 *  | q * [0 r]' * _q |_(1-3) is a vector in the x-y plane corresponding to r
 *  rotated by the angle, theta, in the counter-clockwise direction.
 */
template <typename Scalar, class Exp>
void AxisAngleToQuaternion(const fiber::_RValue<Scalar, Exp>& axis,
                           const Scalar angle, Quaternion<Scalar>* q) {
  Scalar theta_over_2 = angle / Scalar(2.0);
  Scalar cos_theta_over_2 = std::cos(theta_over_2);
  Scalar sin_theta_over_2 = std::sin(theta_over_2);
  (*q)[0] = cos_theta_over_2;
  (*q)[1] = sin_theta_over_2 * axis[0];
  (*q)[2] = sin_theta_over_2 * axis[1];
  (*q)[3] = sin_theta_over_2 * axis[2];
}

/**
 *  We follow the counter-clockwise right hand convention. For example, if r is
 *  a vector in the x-y plane and the axis of rotation is the z-axis, then
 *  | q * [0 r]' * _q |_(1-3) is a vector in the x-y plane corresponding to r
 *  rotated by the angle, theta, in the counter-clockwise direction.
 */
template <typename Scalar>
void AxisAngleToQuaternion(const AxisAngle<Scalar>& axis_angle,
                           Quaternion<Scalar>* q) {
  AxisAngleToQuaternion(axis_angle.GetAxis(), axis_angle.GetAngle(), q);
}

/**
 *  We follow the counter-clockwise right hand convention. For example, if r is
 *  a vector in the x-y plane and the axis of rotation is the z-axis, then
 *  R*r is a vector in the x-y plane corresponding to r rotated by the angle,
 *  theta, in the counter-clockwise direction.
 */
template <typename Scalar, class Exp>
void AxisAngleToRotationMatrix(const AxisAngle<Scalar>& axis_angle,
                               _LValue<Scalar, Exp>* R) {
  Scalar sin_theta = std::sin(axis_angle.GetAngle());
  Scalar cos_theta = std::cos(axis_angle.GetAngle());
  (*R) = cos_theta * Eye<Scalar, 3>()
      + sin_theta * CrossMatrix(axis_angle.GetAxis())
      + (1 - cos_theta) * axis_angle.GetAxis()
          * Transpose(axis_angle.GetAxis());
}

/**
 *  We follow the counter-clockwise right hand convention. For example, if r is
 *  a vector in the x-y plane and the axis of rotation is the z-axis, then
 *  | q * [0 r]' * _q |_(1-3) is a vector in the x-y plane corresponding to r
 *  rotated by the angle, theta, in the counter-clockwise direction.
 */
template <typename Scalar, int Axis>
void CoordinateAxisAngleToQuaternion(
    const CoordinateAxisAngle<Scalar, Axis>& axis_angle,
    Quaternion<Scalar>* q) {
  Scalar theta_over_2 = axis_angle.GetAngle() / Scalar(2.0);
  Scalar cos_theta_over_2 = std::cos(theta_over_2);
  Scalar sin_theta_over_2 = std::sin(theta_over_2);
  (*q)[0] = cos_theta_over_2;
  (*q)[1] = (Axis == 0 ? sin_theta_over_2 : 0);
  (*q)[2] = (Axis == 1 ? sin_theta_over_2 : 0);
  (*q)[3] = (Axis == 2 ? sin_theta_over_2 : 0);
}

/**
 *  We follow the counter-clockwise right hand convention. For example, if r is
 *  a vector in the x-y plane and the axis of rotation is the z-axis, then
 *  R*r is a vector in the x-y plane corresponding to r rotated by the angle,
 *  theta, in the counter-clockwise direction.
 */
template <typename Scalar, int Axis, class Exp>
void CoordinateAxisAngleToRotationMatrix(
    const CoordinateAxisAngle<Scalar, Axis>& axis_angle,
    _LValue<Scalar, Exp>* R) {
  Scalar sin_theta = std::sin(axis_angle.GetAngle());
  Scalar cos_theta = std::cos(axis_angle.GetAngle());
  if(Axis == 0) {
    (*R) << 1,         0,          0,
            0, cos_theta, -sin_theta,
            0, sin_theta,  cos_theta;
  } else if (Axis == 1) {
    (*R) <<  cos_theta, 0, sin_theta,
             0,         1,         0,
            -sin_theta, 0, cos_theta;
  } else if (Axis == 2) {
    (*R) << cos_theta, -sin_theta, 0,
            sin_theta,  cos_theta, 0,
                    0,          0, 1;
  }
}

template <typename Scalar, class Exp>
void QuaternionToAxisAngle(const Quaternion<Scalar>& q,
                           fiber::_LValue<Scalar, Exp>* axis, Scalar* angle) {
  assert(axis->size() >= 3);
  Scalar sin_theta_over_2 = fiber::Norm(fiber::Rows(q, 1, 3));
  Scalar cos_theta_over_2 = q[0];
  *angle = Scalar(2.0) * std::atan2(sin_theta_over_2, cos_theta_over_2);
  /// TODO(bialkowski): need scalar traits to determine this threshold
  if (*angle < 1e-9) {
    *axis = Scalar(1.0) / sin_theta_over_2 * fiber::Rows(q, 1, 3);
  } else {
    *axis = fiber::Rows(q, 1, 3);
  }
}

template <typename Scalar, class Exp>
void QuaternionToRotationMatrix(const Quaternion<Scalar>& q,
                                fiber::_LValue<Scalar, Exp>* R) {
  assert(R->size() >= 9);
  assert(R->rows() >= 3);
  assert(R->cols() >= 3);

  // Note: this is the transpose of what is given in Diebel to match
  // the convention in eigen
  (*R)(0, 0) = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3];
  (*R)(1, 0) = 2 * q[1] * q[2] + 2 * q[0] * q[3];
  (*R)(2, 0) = 2 * q[1] * q[3] - 2 * q[0] * q[2];
  (*R)(0, 1) = 2 * q[1] * q[2] - 2 * q[0] * q[3];
  (*R)(1, 1) = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3];
  (*R)(2, 1) = 2 * q[2] * q[3] + 2 * q[0] * q[1];
  (*R)(0, 2) = 2 * q[1] * q[3] + 2 * q[0] * q[2];
  (*R)(1, 2) = 2 * q[2] * q[3] - 2 * q[0] * q[1];
  (*R)(2, 2) = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3];
}

template <typename Scalar, class Exp>
void RotationMatrixToQuaternion(const fiber::_RValue<Scalar, Exp>& R,
                                Quaternion<Scalar>* q) {
  assert(R.size() >= 9);
  assert(R.rows() >= 3);
  assert(R.cols() >= 3);
  const Scalar r11 = R(0, 0);
  const Scalar r12 = R(0, 1);
  const Scalar r13 = R(0, 2);
  const Scalar r21 = R(1, 0);
  const Scalar r22 = R(1, 1);
  const Scalar r23 = R(1, 2);
  const Scalar r31 = R(2, 0);
  const Scalar r32 = R(2, 1);
  const Scalar r33 = R(2, 2);

  // Note: these are the conjugate of what is given in Diebel to match the
  // convention in eigen
  if (r22 > -r33 && r11 > -r22 && r11 > -r33) {
    const Scalar k = std::sqrt(1 + r11 + r22 + r33);
    (*q)[0] = Scalar(0.5) * k;
    (*q)[1] = -Scalar(0.5) * (r23 - r32) / k;
    (*q)[2] = -Scalar(0.5) * (r31 - r13) / k;
    (*q)[3] = -Scalar(0.5) * (r12 - r21) / k;
  } else if (r22 < -r33 && r11 > r22 && r11 > r33) {
    const Scalar k = std::sqrt(1 + r11 - r22 - r33);
    (*q)[0] = Scalar(0.5) * (r23 - r32) / k;
    (*q)[1] = -Scalar(0.5) * k;
    (*q)[2] = -Scalar(0.5) * (r12 + r21) / k;
    (*q)[3] = -Scalar(0.5) * (r31 + r13) / k;
  } else if (r22 > r33 && r11 < r22 && r11 < -r33) {
    const Scalar k = std::sqrt(1 - r11 + r22 - r33);
    (*q)[0] = Scalar(0.5) * (r31 - r13) / k;
    (*q)[1] = -Scalar(0.5) * (r21 + r12) / k;
    (*q)[2] = -Scalar(0.5) * k;
    (*q)[3] = -Scalar(0.5) * (r23 + r32) / k;
  } else if (r22 < r33 && r11 < -r22 && r11 < r33) {
    const Scalar k = std::sqrt(1 - r11 - r22 + r33);
    (*q)[0] = Scalar(0.5) * (r12 - r21) / k;
    (*q)[1] = -Scalar(0.5) * (r31 + r13) / k;
    (*q)[2] = -Scalar(0.5) * (r23 + r32) / k;
    (*q)[3] = -Scalar(0.5) * k;
  } else {
    assert(false);
  }
}

namespace euler {

enum {
  X = 0,
  Y = 1,
  Z = 2
};

enum {
  _1 = 0,
  _2 = 1,
  _3 = 2
};

/// construction euler angles from quaternion, only certain sequences can
/// be constructed this way
template <typename Scalar, int Axis1, int Axis2, int Axis3>
struct _AnglesFromQuaternion {
  enum { HAS_SPECIALIZATION = false };
};

// Equations from Stanford tech report by James Diebel
template <typename Scalar>
struct _AnglesFromQuaternion<Scalar, _1, _2, _1> {
  enum { HAS_SPECIALIZATION = true };

  static void Build(const Quaternion<Scalar>& q,
                    Angles<Scalar, _1, _2, _1>* angles) {
    (*angles)[0] = std::atan2(2 * q[1] * q[2] - 2 * q[0] * q[3],
                              2 * q[1] * q[3] + 2 * q[0] * q[2]);
    (*angles)[1] =
        std::acos(q[1] * q[1] + q[0] * q[0] - q[3] * q[3] - q[2] * q[2]);
    (*angles)[2] = std::atan2(2 * q[1] * q[2] + 2 * q[0] * q[3],
                              -2 * q[1] * q[3] + 2 * q[0] * q[2]);
  }
};

template <typename Scalar>
struct _AnglesFromQuaternion<Scalar, _1, _2, _3> {
  enum { HAS_SPECIALIZATION = true };

  static void Build(const Quaternion<Scalar>& q,
                    Angles<Scalar, _1, _2, _3>* angles) {
    (*angles)[0] =
        std::atan2(-2 * q[1] * q[2] + 2 * q[0] * q[3],
                   q[1] * q[1] + q[0] * q[0] - q[3] * q[3] - q[2] * q[2]);
    (*angles)[1] = std::asin(2 * q[1] * q[3] + 2 * q[0] * q[2]);
    (*angles)[2] =
        std::atan2(-2 * q[2] * q[3] + 2 * q[0] * q[1],
                   q[3] * q[3] - q[2] * q[2] - q[1] * q[1] + q[0] * q[0]);
  }
};

template <typename Scalar>
struct _AnglesFromQuaternion<Scalar, _3, _2, _1> {
  enum { HAS_SPECIALIZATION = true };

  static void Build(const Quaternion<Scalar>& q,
                    Angles<Scalar, _3, _2, _1>* angles) {
    (*angles)[0] =
        std::atan2(2 * q[2] * q[3] + 2 * q[0] * q[1],
                   q[3] * q[3] - q[2] * q[2] - q[1] * q[1] + q[0] * q[0]);
    (*angles)[1] = -std::asin(2 * q[1] * q[3] - 2 * q[0] * q[2]);
    (*angles)[2] =
        std::atan2(2 * q[1] * q[2] + 2 * q[0] * q[3],
                   q[1] * q[1] + q[0] * q[0] - q[3] * q[3] - q[2] * q[2]);
  }
};

template <typename Scalar>
struct _AnglesFromQuaternion<Scalar, _3, _1, _3> {
  enum { HAS_SPECIALIZATION = true };

  static void Build(const Quaternion<Scalar>& q,
                    Angles<Scalar, _3, _1, _3>* angles) {
    (*angles)[0] = std::atan2(2 * q[1] * q[3] - 2 * q[0] * q[2],
                              2 * q[2] * q[3] + 2 * q[0] * q[1]);
    (*angles)[1] =
        std::acos(q[3] * q[3] - q[2] * q[2] - q[1] * q[1] + q[0] * q[0]);
    (*angles)[2] = std::atan2(2 * q[1] * q[3] + 2 * q[0] * q[2],
                              -2 * q[2] * q[3] + 2 * q[0] * q[1]);
  }
};

}  // namespace euler

template<typename Scalar, int Axis1, int Axis2, int Axis3>
void EulerAnglesToQuaternion(
    const euler::Angles<Scalar, Axis1, Axis2, Axis3>& euler,
    Quaternion<Scalar>* q) {
  *q = CoordinateAxisAngle<Scalar, Axis1>(euler[0]).ToQuaternion()
      * CoordinateAxisAngle<Scalar, Axis2>(euler[1]).ToQuaternion()
      * CoordinateAxisAngle<Scalar, Axis3>(euler[2]).ToQuaternion();
}

}  // namespace fiber



#endif  // FIBER_ROTATION_CONVERSIONS_H_
