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

#ifndef FIBER_PRODUCT_H_
#define FIBER_PRODUCT_H_

#include <fiber/matrix.h>

namespace fiber   {

/// Matrix multiplication
template<typename Scalar, class Exp1, class Exp2>
inline Matrix<Scalar, Exp1::ROWS_, Exp2::COLS_> operator*(
    _RValue<Scalar, Exp1> const& A, _RValue<Scalar, Exp2> const& B) {
  static_assert(Exp1::COLS_ == Exp2::ROWS_,
               "Inner dimensions of matrix multiplication must agree");

  Matrix<Scalar, Exp1::ROWS_, Exp2::COLS_> M;
  for(int i=0; i < Exp1::ROWS_; i++) {
    for(int j=0; j < Exp2::COLS_; j++) {
      M(i,j) = Dot(GetRow(A, i), GetColumn(B, j));
    }
  }
  return M;
}

/// Dot product of two vectors
template <typename Scalar, class Exp1, class Exp2>
inline Scalar Dot(_RValue<Scalar, Exp1> const& A,
                  _RValue<Scalar, Exp2> const& B) {
  static_assert(Exp1::SIZE_ == Exp2::SIZE_,
                "Cannot compute a dot product of vectors that are not the"
                "same size");
  Scalar r(0);
  for (int i = 0; i < A.size(); i++) {
    r += A[i] * B[i];
  }
  return r;
}

}  // namespace fiber

#endif  // FIBER_PRODUCT_H_
