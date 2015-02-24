
#ifndef TEST_FIBER_ASSERT_EQUALS_H_
#define TEST_FIBER_ASSERT_EQUALS_H_

#include <gtest/gtest.h>

template <typename Scalar, class MatA, class MatB>
testing::AssertionResult AssertMatrixEquality(
    const std::string& A_str,
    const std::string& B_str,
    const fiber::_RValue<Scalar, MatA>& A,
    const fiber::_RValue<Scalar, MatB>& B) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    return testing::AssertionFailure()
           << "size(" << A_str << ") != size(" << B_str << ") where:\n"
           << "size " << A_str << ": (" << A.rows() << ", " << A.cols() << ")\n"
           << "size " << B_str << ": (" << B.rows() << ", " << B.cols()
           << ")\n";
  }

  for (int i = 0; i < A.rows(); i++) {
    for (int j = 0; j < A.cols(); j++) {
      if (A(i, j) != B(i, j)) {
        return testing::AssertionFailure()
               << A_str << " != " << B_str << " where\n  " << A_str << ":\n"
               << A << "\n" << B_str << ":\n" << B << "\n";
      }
    }
  }

  return testing::AssertionSuccess();
}

template <typename Scalar, class MatA, class MatB>
testing::AssertionResult AssertMatrixApproxEquality(
    const std::string& A_str,
    const std::string& B_str,
    const std::string& eps_str,
    const fiber::_RValue<Scalar, MatA>& A,
    const fiber::_RValue<Scalar, MatB>& B,
    const Scalar eps) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    return testing::AssertionFailure()
           << "size(" << A_str << ") != size(" << B_str << ") where:\n"
           << "size " << A_str << ": (" << A.rows() << ", " << A.cols() << ")\n"
           << "size " << B_str << ": (" << B.rows() << ", " << B.cols()
           << ")\n";
  }

  for (int i = 0; i < A.rows(); i++) {
    for (int j = 0; j < A.cols(); j++) {
      if ( std::abs(A(i, j) - B(i, j)) > eps) {
        return testing::AssertionFailure()
               << A_str << " != " << B_str
               << " at index (" << i << ", " << j << ") with values "
               << "A(i,j) = " << A(i,j) << ", B(i,j) = " << B(i,j)
               << " where\n  " << A_str << ":\n"
               << A << "\n" << B_str << ":\n" << B << "\n";
      }
    }
  }

  return testing::AssertionSuccess();
}

#endif  // TEST_FIBER_ASSERT_EQUALS_H_
