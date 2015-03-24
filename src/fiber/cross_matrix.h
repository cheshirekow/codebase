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
 *  @date   Dec 12, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef FIBER_CROSS_MATRIX_H_
#define FIBER_CROSS_MATRIX_H_


namespace fiber {
namespace cross_matrix {

/// indices of a vector in it's cross-product representation
const int kIDX[9] = {
    0, 2, 1,
    2, 0, 0,
    1, 0, 0
};

/// signs in the cross-product representation
const int kSIGN[9] = {
    0, -1,  1,
    1,  0, -1,
   -1,  1,  0
};
}  // namespace cross_matrix

/// cross-product matrix of a vector
/**
 *  Given a vector v, the CrossMatrix of v is the skew symmetric 3x3 matrix
 *  given by:
 *  |     0  -v[2]   v[1] |
 *  |  v[2]      0  -v[0] |
 *  | -v[1]   v[0]      0 |
 *
 *  Note: relatively private class, construct with the CrossMatrix function.
 */
template <typename Scalar, class Exp>
class _CrossMatrix : public _RValue<Scalar, _CrossMatrix<Scalar, Exp> > {
 private:
  Exp const& v_;  ///< wrapped expression

 public:
  enum {
    ROWS_ = 3,
    COLS_ = 3,
    SIZE_ = 9
  };

  _CrossMatrix(Exp const& A) : v_(A) {
    static_assert(Exp::SIZE_ == 3,
                  "Cross-product matrices are only defined for size = 3"
                  "vectors");
  }

  Size size() const { return 9; }
  Size rows() const { return 3; }
  Size cols() const { return 3; }

  /// return the evaluated i'th element of a vector expression
  Scalar operator[](Size i) const {
    return (*this)(i / 3, i % 3);
  }

  /// return the evaluated (j,i)'th element of a matrix expression
  Scalar operator()(Size i, Size j) const {
    assert(0 <= i && i <= 2);
    assert(0 <= j && j <= 2);
    return cross_matrix::kSIGN[3 * i + j] * v_[cross_matrix::kIDX[3 * i + j]];
  }
};

/// cross-product matrix of a vector
/**
 *  Given a vector v, the CrossMatrix of v is the skew symmetric 3x3 matrix
 *  given by:
 *  |     0  -v[2]   v[1] |
 *  |  v[2]      0  -v[0] |
 *  | -v[1]   v[0]      0 |
 */
template <typename Scalar, class Exp>
inline _CrossMatrix<Scalar, Exp> CrossMatrix(_RValue<Scalar, Exp> const& A) {
  return _CrossMatrix<Scalar, Exp>(static_cast<Exp const&>(A));
}

}  // namespace fiber

#endif  // FIBER_CROSS_MATRIX_H_
