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
 *  @date   Dec 09, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */


#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <fiber/fiber.h>

#include "assert_equals.h"

using namespace fiber;

TEST(QuaternionTest, ManualTests) {
  fiber::Quaternion<double> q1(1,0,0,0);
  EXPECT_PRED_FORMAT2(AssertMatrixEquality,
                      q1*q1,q1);

  // rotate the x-axis by angle, about the y-axis
  double angle = M_PI/10;
  EXPECT_PRED_FORMAT3(
      AssertMatrixApproxEquality,
      (fiber::Matrix<double, 3, 1>(std::cos(angle), 0, -std::sin(angle))),
      (fiber::CoordinateAxisAngle<double, 1>(angle).ToQuaternion().Rotate(
          fiber::CoordinateAxis<double, 0>())),
      1e-9);

  // rotate the z-axis by angle, about the y-axis
  EXPECT_PRED_FORMAT3(
  AssertMatrixApproxEquality,
  (fiber::Matrix<double, 3, 1>(std::sin(angle), 0, std::cos(angle))),
  (fiber::CoordinateAxisAngle<double, 1>(angle).ToQuaternion().Rotate(
      fiber::CoordinateAxis<double, 2>())),
  1e-9);

  angle = -M_PI/10;
  EXPECT_PRED_FORMAT3(
      AssertMatrixApproxEquality,
      (fiber::Matrix<double, 3, 1>(std::cos(angle), 0, -std::sin(angle))),
      (fiber::CoordinateAxisAngle<double, 1>(angle).ToQuaternion().Rotate(
          fiber::CoordinateAxis<double, 0>())),
      1e-9);
}

::testing::AssertionResult QuaternionFromAxisAngleMatchesEigen(
    const fiber::AxisAngle<double>& aa_fiber) {
  Eigen::Vector3d axis(aa_fiber.GetAxis()[0], aa_fiber.GetAxis()[1],
                       aa_fiber.GetAxis()[2]);
  Eigen::AngleAxisd aa_eigen(aa_fiber.GetAngle(), axis);
  Eigen::Quaterniond q_eigen(aa_eigen);

  fiber::Quaternion<double> q_fiber = aa_fiber;
  fiber::Quaternion<double> q_from_eigen(q_eigen.w(), q_eigen.x(), q_eigen.y(),
                                          q_eigen.z());
  return AssertMatrixApproxEquality("q_fiber", "q_eigen", "eps", q_fiber,
                                    q_from_eigen, 1e-9);
}

TEST(EigenCompareTest, QuaternionFromAxisAngleMatchesEigenTest) {
  fiber::AxisAngled aa_fiber;
  aa_fiber.SetAxis(fiber::Vector3d(1, 0, 0));
  aa_fiber.SetAngle(M_PI);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_fiber));
  aa_fiber.SetAngle(M_PI / 2);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_fiber));
  aa_fiber.SetAngle(M_PI / 4);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_fiber));

  aa_fiber.SetAxis(fiber::Normalize(fiber::Vector3d(1, 1, 1)));
  aa_fiber.SetAngle(M_PI);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_fiber));
  aa_fiber.SetAngle(M_PI / 2);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_fiber));
  aa_fiber.SetAngle(M_PI / 4);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_fiber));
}

template <typename Scalar>
::testing::AssertionResult QuaternionRotationMatchesEigen(
    const fiber::Quaternion<Scalar>& q_fiber,
    const fiber::Matrix<Scalar, 3, 1>& r_fiber) {
  Eigen::Quaternion<Scalar> q_eigen(q_fiber[0], q_fiber[1], q_fiber[2],
                                    q_fiber[3]);

  Eigen::Matrix<Scalar, 3, 1> r_eigen(r_fiber[0], r_fiber[1], r_fiber[2]);
  Eigen::Matrix<Scalar, 3, 1> rp_eigen = q_eigen * r_eigen;
  fiber::Matrix<Scalar, 3, 1> rp_fiber = q_fiber.Rotate(r_fiber);
  fiber::Matrix<Scalar, 3, 1> rp_from_eigen(rp_eigen[0], rp_eigen[1],
                                             rp_eigen[2]);

  return AssertMatrixApproxEquality("rp_fiber", "rp_eigen", "eps", rp_fiber,
                                    rp_from_eigen, 1e-9);
}

template <typename Scalar, class Exp>
::testing::AssertionResult QuaternionToRotationMatrixMatchesRotation(
    const fiber::Quaternion<Scalar>& q,
    const fiber::_RValue<Scalar, Exp>& r) {
  fiber::Matrix<Scalar, 3, 3> R;
  QuaternionToRotationMatrix(q, &R);
  return AssertMatrixApproxEquality("q*r*q'", "R(q)*r", "eps", q.Rotate(r),
                                    R * r, 1e-9);
}

TEST(EigenCompareTest, QuaternionRotationMatchesEigenTest) {
  fiber::Quaternion<double> q;
  fiber::Matrix<double,3,1> r(1, 0, 0);

  fiber::Matrix<double, 3, 1> axis =
      fiber::Normalize(fiber::Matrix<double, 3, 1>(1, 1, 1));
  q = fiber::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionRotationMatchesEigen(q,r));
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesRotation(q,r));

  q = fiber::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionRotationMatchesEigen(q,r));
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesRotation(q,r));

  axis = fiber::Normalize(fiber::Matrix<double, 3, 1>(1, 2, 3));
  q = fiber::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionRotationMatchesEigen(q,r));
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesRotation(q,r));

  q = fiber::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionRotationMatchesEigen(q,r));
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesRotation(q,r));
}

template <typename Scalar>
::testing::AssertionResult QuaternionToRotationMatrixMatchesEigen(
    const fiber::Quaternion<Scalar>& q_fiber) {
  Eigen::Quaternion<Scalar> q_eigen(q_fiber[0], q_fiber[1], q_fiber[2],
                                    q_fiber[3]);
  Eigen::Matrix<Scalar, 3, 3> R_eigen = q_eigen.matrix();
  fiber::Matrix<Scalar, 3, 3> R_from_eigen;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R_from_eigen(i, j) = R_eigen(i, j);
    }
  }

  fiber::Matrix<Scalar, 3, 3> R_fiber;
  QuaternionToRotationMatrix(q_fiber, &R_fiber);
  return AssertMatrixApproxEquality("R_fiber", "R_eigen", "eps", R_fiber,
                                    R_from_eigen, 1e-9);
}

TEST(EigenCompareTest, QuaternionToRotationMatchesEigenTest) {
  fiber::Quaternion<double> q;

  fiber::Matrix<double, 3, 1> axis =
        fiber::Normalize(fiber::Matrix<double, 3, 1>(1, 1, 1));
  q = fiber::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesEigen(q));

  q = fiber::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesEigen(q));

  axis = fiber::Normalize(fiber::Matrix<double, 3, 1>(1, 2, 3));
  q = fiber::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesEigen(q));

  q = fiber::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesEigen(q));
}

template <typename Scalar>
::testing::AssertionResult QuaternionToRotationMatrixIsInvolute(
    const fiber::Quaternion<Scalar>& q) {
  fiber::Matrix<Scalar, 3, 3> R;
  fiber::Quaternion<Scalar> q_R;
  QuaternionToRotationMatrix(q, &R);
  RotationMatrixToQuaternion(R, &q_R);
  return AssertMatrixApproxEquality("q", "q(R(q))", "eps", q, q_R, 1e-9);
}

TEST(EigenCompareTest, QuaternionToRotationMatrixIsInvoluteTest) {
  fiber::Quaternion<double> q;

  fiber::Matrix<double, 3, 1> axis =
        fiber::Normalize(fiber::Matrix<double, 3, 1>(1, 1, 1));
  q = fiber::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionToRotationMatrixIsInvolute(q));

  q = fiber::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionToRotationMatrixIsInvolute(q));

  axis = fiber::Normalize(fiber::Matrix<double, 3, 1>(1, 2, 3));
  q = fiber::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionToRotationMatrixIsInvolute(q));

  q = fiber::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionToRotationMatrixIsInvolute(q));
}
