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
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef FIBER_NORMALIZE_H_
#define FIBER_NORMALIZE_H_

#include <cmath>
#include <fiber/product.h>

namespace fiber {

/// Return the squared 2-norm of a vector
template <typename Scalar, class Exp>
inline Scalar SquaredNorm(_RValue<Scalar, Exp> const& A) {
  return Dot(A,A);
}

/// Return the 2-norm of a vector
template <typename Scalar, class Exp>
inline Scalar Norm(_RValue<Scalar, Exp> const& A) {
  return std::sqrt(SquaredNorm(A));
}

/// Return the n-norm of a vector
template <typename Scalar, class Exp>
inline Scalar Norm(_RValue<Scalar, Exp> const& A, int n) {
  Scalar r(0);
  for (int i = 0; i < A.size(); i++) {
    r += std::pow(A[i], n);
  }
  return std::pow(r, 1.0/n);
}

/// Expression template presents a view of a matrix where each element is
/// normalized, such that the sum of the elements is 1.0
template <typename Scalar, class Exp>
class _Normalize : public _RValue<Scalar, _Normalize<Scalar, Exp> > {
  Scalar norm_;     ///< 2-norm of elements in the expression
  Exp const& v_;    ///< underlying expression

 public:
  enum {
    ROWS_ = Exp::ROWS_,
    COLS_ = Exp::COLS_,
    SIZE_ = ROWS_ * COLS_
  };

  _Normalize(Exp const& A) : norm_(Norm(A)), v_(A) {}

  Size size() const { return v_.size(); }
  Size rows() const { return v_.rows(); }
  Size cols() const { return v_.cols(); }

  Scalar operator[](Index i) const {
    return v_[i] / norm_;
  }

  Scalar operator()(Index i, Index j) const {
    return v_(i, j) / norm_;
  }
};

/// Prresents a view of a matrix or vector where each element is
/// normalized, such that the sum of the elements is 1.0
template <typename Scalar, class Exp>
inline _Normalize<Scalar, Exp> Normalize(_RValue<Scalar, Exp> const& A) {
  return _Normalize<Scalar, Exp>(static_cast<Exp const&>(A));
}

}  // namespace fiber


#endif  // FIBER_NORMALIZE_H_
