/*
 *  Copyright (C) 2014 Josh Bialkowski (jbialk@mit.edu)
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
 *  @date   Dec 09, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */


#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mpblocks/linalg.h>

#include "assert_equals.h"

using namespace mpblocks;

TEST(QuaternionTest, ManualTests) {
  linalg::Quaternion<double> q1(1,0,0,0);
  EXPECT_PRED_FORMAT2(AssertMatrixEquality,
                      q1*q1,q1);

  // rotate the x-axis by angle, about the y-axis
  double angle = M_PI/10;
  EXPECT_PRED_FORMAT3(
      AssertMatrixApproxEquality,
      (linalg::Matrix<double, 3, 1>(std::cos(angle), 0, -std::sin(angle))),
      (linalg::CoordinateAxisAngle<double, 1>(angle).ToQuaternion().Rotate(
          linalg::CoordinateAxis<double, 0>())),
      1e-9);

  // rotate the z-axis by angle, about the y-axis
  EXPECT_PRED_FORMAT3(
  AssertMatrixApproxEquality,
  (linalg::Matrix<double, 3, 1>(std::sin(angle), 0, std::cos(angle))),
  (linalg::CoordinateAxisAngle<double, 1>(angle).ToQuaternion().Rotate(
      linalg::CoordinateAxis<double, 2>())),
  1e-9);

  angle = -M_PI/10;
  EXPECT_PRED_FORMAT3(
      AssertMatrixApproxEquality,
      (linalg::Matrix<double, 3, 1>(std::cos(angle), 0, -std::sin(angle))),
      (linalg::CoordinateAxisAngle<double, 1>(angle).ToQuaternion().Rotate(
          linalg::CoordinateAxis<double, 0>())),
      1e-9);
}

::testing::AssertionResult QuaternionFromAxisAngleMatchesEigen(
    const linalg::AxisAngle<double>& aa_linalg) {
  Eigen::Vector3d axis(aa_linalg.GetAxis()[0], aa_linalg.GetAxis()[1],
                       aa_linalg.GetAxis()[2]);
  Eigen::AngleAxisd aa_eigen(aa_linalg.GetAngle(), axis);
  Eigen::Quaterniond q_eigen(aa_eigen);

  linalg::Quaternion<double> q_linalg = aa_linalg;
  linalg::Quaternion<double> q_from_eigen(q_eigen.w(), q_eigen.x(), q_eigen.y(),
                                          q_eigen.z());
  return AssertMatrixApproxEquality("q_linalg", "q_eigen", "eps", q_linalg,
                                    q_from_eigen, 1e-9);
}

TEST(EigenCompareTest, QuaternionFromAxisAngleMatchesEigenTest) {
  linalg::AxisAngled aa_linalg;
  aa_linalg.SetAxis(linalg::Vector3d(1, 0, 0));
  aa_linalg.SetAngle(M_PI);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_linalg));
  aa_linalg.SetAngle(M_PI / 2);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_linalg));
  aa_linalg.SetAngle(M_PI / 4);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_linalg));

  aa_linalg.SetAxis(linalg::Normalize(linalg::Vector3d(1, 1, 1)));
  aa_linalg.SetAngle(M_PI);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_linalg));
  aa_linalg.SetAngle(M_PI / 2);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_linalg));
  aa_linalg.SetAngle(M_PI / 4);
  EXPECT_TRUE(QuaternionFromAxisAngleMatchesEigen(aa_linalg));
}

template <typename Scalar>
::testing::AssertionResult QuaternionRotationMatchesEigen(
    const linalg::Quaternion<Scalar>& q_linalg,
    const linalg::Matrix<Scalar, 3, 1>& r_linalg) {
  Eigen::Quaternion<Scalar> q_eigen(q_linalg[0], q_linalg[1], q_linalg[2],
                                    q_linalg[3]);

  Eigen::Matrix<Scalar, 3, 1> r_eigen(r_linalg[0], r_linalg[1], r_linalg[2]);
  Eigen::Matrix<Scalar, 3, 1> rp_eigen = q_eigen * r_eigen;
  linalg::Matrix<Scalar, 3, 1> rp_linalg = q_linalg.Rotate(r_linalg);
  linalg::Matrix<Scalar, 3, 1> rp_from_eigen(rp_eigen[0], rp_eigen[1],
                                             rp_eigen[2]);

  return AssertMatrixApproxEquality("rp_linalg", "rp_eigen", "eps", rp_linalg,
                                    rp_from_eigen, 1e-9);
}

template <typename Scalar, class Exp>
::testing::AssertionResult QuaternionToRotationMatrixMatchesRotation(
    const linalg::Quaternion<Scalar>& q,
    const linalg::_RValue<Scalar, Exp>& r) {
  linalg::Matrix<Scalar, 3, 3> R;
  QuaternionToRotationMatrix(q, &R);
  return AssertMatrixApproxEquality("q*r*q'", "R(q)*r", "eps", q.Rotate(r),
                                    R * r, 1e-9);
}

TEST(EigenCompareTest, QuaternionRotationMatchesEigenTest) {
  linalg::Quaternion<double> q;
  linalg::Matrix<double,3,1> r(1, 0, 0);

  linalg::Matrix<double, 3, 1> axis =
      linalg::Normalize(linalg::Matrix<double, 3, 1>(1, 1, 1));
  q = linalg::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionRotationMatchesEigen(q,r));
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesRotation(q,r));

  q = linalg::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionRotationMatchesEigen(q,r));
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesRotation(q,r));

  axis = linalg::Normalize(linalg::Matrix<double, 3, 1>(1, 2, 3));
  q = linalg::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionRotationMatchesEigen(q,r));
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesRotation(q,r));

  q = linalg::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionRotationMatchesEigen(q,r));
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesRotation(q,r));
}

template <typename Scalar>
::testing::AssertionResult QuaternionToRotationMatrixMatchesEigen(
    const linalg::Quaternion<Scalar>& q_linalg) {
  Eigen::Quaternion<Scalar> q_eigen(q_linalg[0], q_linalg[1], q_linalg[2],
                                    q_linalg[3]);
  Eigen::Matrix<Scalar, 3, 3> R_eigen = q_eigen.matrix();
  linalg::Matrix<Scalar, 3, 3> R_from_eigen;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R_from_eigen(i, j) = R_eigen(i, j);
    }
  }

  linalg::Matrix<Scalar, 3, 3> R_linalg;
  QuaternionToRotationMatrix(q_linalg, &R_linalg);
  return AssertMatrixApproxEquality("R_linalg", "R_eigen", "eps", R_linalg,
                                    R_from_eigen, 1e-9);
}

TEST(EigenCompareTest, QuaternionToRotationMatchesEigenTest) {
  linalg::Quaternion<double> q;

  linalg::Matrix<double, 3, 1> axis =
        linalg::Normalize(linalg::Matrix<double, 3, 1>(1, 1, 1));
  q = linalg::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesEigen(q));

  q = linalg::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesEigen(q));

  axis = linalg::Normalize(linalg::Matrix<double, 3, 1>(1, 2, 3));
  q = linalg::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesEigen(q));

  q = linalg::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionToRotationMatrixMatchesEigen(q));
}

template <typename Scalar>
::testing::AssertionResult QuaternionToRotationMatrixIsInvolute(
    const linalg::Quaternion<Scalar>& q) {
  linalg::Matrix<Scalar, 3, 3> R;
  linalg::Quaternion<Scalar> q_R;
  QuaternionToRotationMatrix(q, &R);
  RotationMatrixToQuaternion(R, &q_R);
  return AssertMatrixApproxEquality("q", "q(R(q))", "eps", q, q_R, 1e-9);
}

TEST(EigenCompareTest, QuaternionToRotationMatrixIsInvoluteTest) {
  linalg::Quaternion<double> q;

  linalg::Matrix<double, 3, 1> axis =
        linalg::Normalize(linalg::Matrix<double, 3, 1>(1, 1, 1));
  q = linalg::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionToRotationMatrixIsInvolute(q));

  q = linalg::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionToRotationMatrixIsInvolute(q));

  axis = linalg::Normalize(linalg::Matrix<double, 3, 1>(1, 2, 3));
  q = linalg::AxisAngle<double>(axis, M_PI/4);
  EXPECT_TRUE(QuaternionToRotationMatrixIsInvolute(q));

  q = linalg::AxisAngle<double>(axis, M_PI/2);
  EXPECT_TRUE(QuaternionToRotationMatrixIsInvolute(q));
}
