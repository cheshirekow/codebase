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
 *  @date   Dec 12, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef MPBLOCKS_LINALG_CROSS_MATRIX_H_
#define MPBLOCKS_LINALG_CROSS_MATRIX_H_

namespace mpblocks {
namespace linalg {

namespace cross_matrix {

const int kIDX[9] = {
    0, 2, 1,
    2, 0, 0,
    1, 0, 0
};

const int kSIGN[9] = {
    0, -1,  1 ,
    1,  0, -1 ,
   -1,  1,  0
};

}

/// cross-product matrix of a vector
/**
 *  Given a vector v, the CrossMatrix of v is the skew symmetric 3x3 matrix
 *  given by:
 *  |     0  -v[2]   v[1] |
 *  |  v[2]      0  -v[0] |
 *  | -v[1]   v[0]      0 |
 */
template <typename Scalar, class Exp>
class _CrossMatrix : public _RValue<Scalar, _CrossMatrix<Scalar, Exp> > {
 private:
  Exp const& v_;

 public:
  typedef unsigned int Size_t;

  _CrossMatrix(Exp const& A) : v_(A) {
    assert(v_.size() == 3);
  }

  Size_t size() const { return 9; }
  Size_t rows() const { return 3; }
  Size_t cols() const { return 3; }

  /// return the evaluated i'th element of a vector expression
  Scalar operator[](Size_t i) const {
    return (*this)(i / 3, i % 3);
  }

  /// return the evaluated (j,i)'th element of a matrix expression
  Scalar operator()(Size_t i, Size_t j) const {
    assert(0 <= i && i <= 2);
    assert(0 <= j && j <= 2);
    return cross_matrix::kSIGN[3 * i + j] * v_[cross_matrix::kIDX[3 * i + j]];
  }
};

template <typename Scalar, class Exp>
inline _CrossMatrix<Scalar, Exp> CrossMatrix(_RValue<Scalar, Exp> const& A) {
  return _CrossMatrix<Scalar, Exp>(static_cast<Exp const&>(A));
}

}  // namespace linalg
}  // namespace mpblocks

#endif  // MPBLOCKS_LINALG_CROSS_MATRIX_H_
