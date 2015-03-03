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
 *  @date   Dec 9, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef FIBER_AXIS_ANGLE_H_
#define FIBER_AXIS_ANGLE_H_

namespace fiber {

/// Encodes a rotation in 3-dimensions by an return _RView<Scalar, Exp, rows, Exp::COLS_>(static_cast<Exp const&>(A), i, 0);axis of rotation and an angle
/// magnitude of rotation about that axis.
template <typename Scalar>
class AxisAngle {
 public:
  typedef Matrix<Scalar, 3, 1> AxisType;

  AxisAngle() : axis_(1, 0, 0), angle_(0) {}

  template <typename Derived>
  AxisAngle(const _RValue<Scalar, Derived>& axis, Scalar angle)
      : axis_(axis), angle_(angle) {}

  AxisAngle(const Quaternion<Scalar>& q) {
    QuaternionToAxisAngle(q, &axis_, &angle_);
  }

  const AxisType& GetAxis() const {
    return axis_;
  }

  template <class Exp>
  void SetAxis(const _RValue<Scalar, Exp>& axis) {
    static_assert(Exp::SIZE_ == 3,
                  "Cannot set an axis from a vector with size other than 3");
    LValue(axis_) = axis;
  }

  Scalar GetAngle() const {
    return angle_;
  }

  void SetAngle(const Scalar angle) {
    angle_ = angle;
  }

 private:
  AxisType axis_;
  Scalar angle_;
};

/// A 3x1 normal vector with one unity element
template <typename Scalar, int Axis>
class CoordinateAxis
    : public _RValue<Scalar, CoordinateAxis<Scalar, Axis> > {
 public:
  enum {
    SIZE_ = 3,
    ROWS_ = 3,
    COLS_ = 1
  };

  Size size() const { return 3; }
  Size rows() const { return 3; }
  Size cols() const { return 1; }

  CoordinateAxis() {
    static_assert(0 <= Axis && Axis < 3,
                  "A primitive axis must have axis 0, 1, or 2");
  }

  Scalar operator[](Index i) const {
    return (i == Axis) ? Scalar(1.0) : Scalar(0.0);
  }

  Scalar operator()(Index i, Index j) const {
    assert(j == 0);
    return (i == Axis) ? Scalar(1.0) : Scalar(0.0);
  }
};

/// An axis angle rotation about a coordinate axis
template <typename Scalar, int Axis>
class CoordinateAxisAngle {
 public:
  CoordinateAxisAngle() :
    angle_(0) {
  }

  CoordinateAxisAngle(Scalar angle) :
    angle_(angle) {
  }

  Quaternion<Scalar> ToQuaternion() const {
    Quaternion<Scalar> q;
    CoordinateAxisAngleToQuaternion(*this, &q);
    return q;
  }

  CoordinateAxis<Scalar, Axis> GetAxis() const {
    return CoordinateAxis<Scalar, Axis>();
  }

  Scalar GetAngle() const {
    return angle_;
  }

  void SetAngle(Scalar angle) {
    angle_ = angle;
  }

 private:
  Scalar angle_;
};

typedef AxisAngle<double> AxisAngled;
typedef AxisAngle<float> AxisAnglef;

typedef CoordinateAxis<double,0> AxisXd;
typedef CoordinateAxis<double,1> AxisYd;
typedef CoordinateAxis<double,2> AxisZd;

typedef CoordinateAxis<float,0> AxisXf;
typedef CoordinateAxis<float,1> AxisYf;
typedef CoordinateAxis<float,2> AxisZf;

typedef CoordinateAxisAngle<double,0> CoordinateAxisAngleXd;
typedef CoordinateAxisAngle<double,1> CoordinateAxisAngleYd;
typedef CoordinateAxisAngle<double,2> CoordinateAxisAngleZd;

typedef CoordinateAxisAngle<float,0> CoordinateAxisAngleXf;
typedef CoordinateAxisAngle<float,1> CoordinateAxisAngleYf;
typedef CoordinateAxisAngle<float,2> CoordinateAxisAngleZf;

}  // namespace fiber

#endif  // FIBER_AXISANGLE_H_
