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

TEST(EulerAnglesSimpleTest, ManualTests) {
  linalg::Quaterniond q_expected(0, 1, 0, 0);
  linalg::Quaterniond q_actual =
      linalg::euler::Angles<double, 0, 1, 2>(M_PI, 0, 0);
  EXPECT_PRED_FORMAT3(AssertMatrixApproxEquality, q_expected, q_actual, 1e-16);
}

template <typename Scalar_, int Axis1, int Axis2, int Axis3>
struct TestSpec {
  typedef Scalar_ Scalar;
  enum {
    _1 = Axis1,
    _2 = Axis2,
    _3 = Axis3
  };
};

template <typename Scalar, int... Axes>
::testing::AssertionResult AssertQuaternionInvolution(
    const linalg::euler::Angles<Scalar, Axes...>& euler) {
  linalg::Quaternion<Scalar> q = euler;
  linalg::euler::Angles<Scalar, Axes...> q_euler(q);
  return AssertMatrixApproxEquality("euler", "euler(q(euler))", "eps", euler,
                                    q_euler, 1e-8)
         << "and q =\n" << q << "\nnorm(q): " << linalg::Norm(q) << "\n";
}

template <typename Scalar, int Axis>
Eigen::Matrix<Scalar,3,1> EigenCoordinateAxis() {
  static_assert(0 <= Axis && Axis <= 2, "Axis must be in [0,2]");
  if(Axis == 0) {
    return Eigen::Matrix<Scalar,3,1>::UnitX();
  } else if(Axis == 1) {
    return Eigen::Matrix<Scalar,3,1>::UnitY();
  } else if(Axis == 2) {
    return Eigen::Matrix<Scalar,3,1>::UnitZ();
  }
  return  Eigen::Matrix<Scalar,3,1>::UnitX();
}

template <typename Scalar, int... Axes>
::testing::AssertionResult AssertQuaternionMatchesEigen(
    const linalg::euler::Angles<Scalar, Axes...>& euler) {
  typedef TestSpec<Scalar, Axes...> ThisSpec;
  linalg::Quaternion<Scalar> q = euler;
  Eigen::Quaternion<Scalar> q_eigen =
      Eigen::AngleAxis<Scalar>(euler[0],
                               EigenCoordinateAxis<Scalar, ThisSpec::_1>()) *
      Eigen::AngleAxis<Scalar>(euler[1],
                               EigenCoordinateAxis<Scalar, ThisSpec::_2>()) *
      Eigen::AngleAxis<Scalar>(euler[2],
                               EigenCoordinateAxis<Scalar, ThisSpec::_3>());

  linalg::Quaternion<Scalar> q_from_eigen(q_eigen.w(), q_eigen.x(), q_eigen.y(),
                                          q_eigen.z());
  return AssertMatrixApproxEquality("q(euler)", "q_eigen", "eps", q,
                                    q_from_eigen, 1e-8)
         << "and norm(q): " << linalg::Norm(q) << "\n";
}

// Note: Euler angles where one of the axes is repeated aren't quite unique
// so the euler angle -> quaternion transformation may not be involute
template <class TestSpec>
class EulerAnglesTest : public ::testing::Test {};
TYPED_TEST_CASE_P(EulerAnglesTest);
TYPED_TEST_P(EulerAnglesTest, QuaternionInvolution) {
  linalg::euler::Angles<typename TypeParam::Scalar, TypeParam::_1,
                        TypeParam::_2, TypeParam::_3> euler;
  euler << M_PI/10.0, M_PI/10.0, M_PI/10.0;
  EXPECT_TRUE(AssertQuaternionInvolution(euler));
  EXPECT_TRUE(AssertQuaternionMatchesEigen(euler));

//  euler << -M_PI/10.0, M_PI/10.0, M_PI/10.0;
//  EXPECT_TRUE(AssertQuaternionInvolution(euler));
//  EXPECT_TRUE(AssertQuaternionMatchesEigen(euler));
//
//  euler << -M_PI/10.0, M_PI/10.0, -M_PI/10.0;
//  EXPECT_TRUE(AssertQuaternionInvolution(euler));
//  EXPECT_TRUE(AssertQuaternionMatchesEigen(euler));
//
//  euler << M_PI/4.0, M_PI/4.0, M_PI/4.0;
//  EXPECT_TRUE(AssertQuaternionInvolution(euler));
//  EXPECT_TRUE(AssertQuaternionMatchesEigen(euler));
}

REGISTER_TYPED_TEST_CASE_P(EulerAnglesTest, QuaternionInvolution);

using namespace linalg::euler;
typedef ::testing::Types<TestSpec<double, _1, _2, _1>,
                         TestSpec<double, _1, _2, _3>,
                         TestSpec<double, _3, _2, _1>,
                         TestSpec<double, _3, _1, _3> > TestTypes;
INSTANTIATE_TYPED_TEST_CASE_P(My, EulerAnglesTest, TestTypes);

