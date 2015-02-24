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
 *  @date   Oct 17, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef MPBLOCKS_LINALG_QUATERNION_H_
#define MPBLOCKS_LINALG_QUATERNION_H_

namespace mpblocks {
namespace linalg {

template <typename Scalar>
class Quaternion : public linalg::_RValue<Scalar, Quaternion<Scalar> > {
 public:
  typedef unsigned int Size_t;

 private:
  Scalar data_[4];

 public:
  Size_t size() const { return 4; }
  Size_t rows() const { return 4; }
  Size_t cols() const { return 1; }

  Scalar& w() { return data_[0]; }
  Scalar w() const { return data_[0]; }

  Scalar& x() { return data_[1]; }
  Scalar x() const { return data_[1]; }

  Scalar& y() { return data_[2]; }
  Scalar y() const { return data_[2]; }

  Scalar& z() { return data_[3]; }
  Scalar z() const { return data_[3]; }

  Scalar& operator[](Size_t i) {
    assert(0 <= i && i < 4);
    return data_[i];
  }

  Scalar operator[](Size_t i) const {
    assert(0 <= i && i < 4);
    return data_[i];
  }

  Scalar operator()(Size_t i, Size_t j) const {
    assert(0 <= i && i < 4);
    assert(j == 0);
    return data_[i];
  }

  /// Default constructor, identity quaternion
  Quaternion() {
    data_[0] = 1;
    for (int i = 0; i < 3; i++) {
      data_[1 + i] = 0;
    }
  }

  /// inline constructor
  Quaternion(Scalar w, Scalar x, Scalar y, Scalar z) {
    data_[0] = w;
    data_[1] = x;
    data_[2] = y;
    data_[3] = z;
  }

  /// Construct from any MatrixExpression, copies elements
  template <typename Exp>
  Quaternion(const linalg::_RValue<Scalar, Exp>& exp) {
    assert(exp.size() == 4);
    for (int i = 0; i < 4; i++) {
      data_[i] = exp[i];
    }
  }

  Quaternion(const AxisAngle<Scalar>& axis_angle) {
    AxisAngleToQuaternion(axis_angle, this);
  }

  template <int Axis>
  Quaternion(const CoordinateAxisAngle<Scalar, Axis>& axis_angle) {
    CoordinateAxisAngleToQuaternion(axis_angle, this);
  }

  template <int... Axes>
  Quaternion(const euler::Angles<Scalar, Axes...>& euler) {
    EulerAnglesToQuaternion(euler, this);
  }

  template <typename Exp>
  Quaternion<Scalar>& operator=(const linalg::_RValue<Scalar, Exp>& exp) {
    assert(exp.size() == 4);
    for (int i = 0; i < 4; i++) {
      data_[i] = exp[i];
    }
    return *this;
  }

  Quaternion<Scalar>& operator=(const AxisAngle<Scalar>& axis_angle) {
    AxisAngleToQuaternion(axis_angle, this);
    return *this;
  }

  template <int Axis>
  Quaternion<Scalar>& operator=(
      const CoordinateAxisAngle<Scalar, Axis>& axis_angle) {
    CoordinateAxisAngleToQuaternion(axis_angle, this);
    return *this;
  }

  template <int... Axes>
  Quaternion<Scalar>& operator=(const euler::Angles<Scalar, Axes...>& euler) {
    EulerAnglesToQuaternion(euler, this);
    return *this;
  }

  template <typename Exp>
  Matrix<Scalar, 3, 1> Rotate(const linalg::_RValue<Scalar, Exp>& exp) const {
    assert(exp.size() == 3);
    Quaternion<Scalar> projected(0, exp[0], exp[1], exp[2]);
    return linalg::ViewRows((*this) * projected * Conjugate(*this), 1, 3);
  }
};

template <typename Scalar>
inline Quaternion<Scalar> Conjugate(Quaternion<Scalar> const& q) {
  return Quaternion<Scalar>(q[0], -q[1], -q[2], -q[3]);
}

/// We define quaternion multiplication as in Diebel
/**
 *  The quaternion multiplication q*p is equivalent to the matrix left-multiply
 *  of Q(q)*p where Q(q) =
 *  @verbatim
 *   | q0 -q1 -q2 -q3 |
 *   | q1  q0 -q3  q2 |
 *   | q2  q3  q0 -q1 |
 *   | q3 -q2  q1  q0 |
 *@endverbatim
 */
template <typename Scalar>
inline Quaternion<Scalar> operator*(Quaternion<Scalar> const& q_A,
                                    Quaternion<Scalar> const& q_B) {
  return Quaternion<Scalar>(
      q_A[0] * q_B[0] - q_A[1] * q_B[1] - q_A[2] * q_B[2] - q_A[3] * q_B[3],
      q_A[1] * q_B[0] + q_A[0] * q_B[1] - q_A[3] * q_B[2] + q_A[2] * q_B[3],
      q_A[2] * q_B[0] + q_A[3] * q_B[1] + q_A[0] * q_B[2] - q_A[1] * q_B[3],
      q_A[3] * q_B[0] - q_A[2] * q_B[1] + q_A[1] * q_B[2] + q_A[0] * q_B[3]);
// Same code as above but reordered with the A terms in order, to help
// verify correctness
// return Quaternion<Scalar>(
//      q_A[0] * q_B[0] - q_A[1] * q_B[1] - q_A[2] * q_B[2] - q_A[3] * q_B[3],
//      q_A[0] * q_B[1] + q_A[1] * q_B[0] + q_A[2] * q_B[3] - q_A[3] * q_B[2],
//      q_A[0] * q_B[2] - q_A[1] * q_B[3] + q_A[2] * q_B[0] + q_A[3] * q_B[1],
//      q_A[0] * q_B[3] + q_A[1] * q_B[2] - q_A[2] * q_B[1] + q_A[3] * q_B[0]);
}

typedef Quaternion<double> Quaterniond;
typedef Quaternion<float> Quaternionf;

}  // namespace linalg
}  // namespace mpblocks

#endif  // MPBLOCKS_LINALG_QUATERNION_H_
