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
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef FIBER_MATRIX_H_
#define FIBER_MATRIX_H_


namespace fiber {

template <typename Scalar, int Rows, int Cols>
class Matrix : public _LValue<Scalar, Matrix<Scalar, Rows, Cols> >,
               public _RValue<Scalar, Matrix<Scalar, Rows, Cols> > {

 protected:
  Scalar data_[Rows * Cols];

 public:
  enum {
    ROWS_ = Rows,
    COLS_ = Cols,
    SIZE_ = Rows * Cols
  };

  Size size() const { return Rows * Cols; }
  Size rows() const { return Rows; }
  Size cols() const { return Cols; }

  Scalar& operator[](Index i) {
    assert(i < SIZE_);
    return data_[i];
  }

  Scalar const& operator[](Index i) const {
    assert(i < SIZE_);
    return data_[i];
  }

  Scalar& operator()(int i, int j) {
    assert(i < Rows && j < Cols);
    return data_[i * Cols + j];
  }

  Scalar operator()(int i, int j) const {
    assert(i < Rows && j < Cols);
    return data_[i * Cols + j];
  }

  /// Default constructor
  Matrix() {}

  /// Construct from any MatrixExpression:
  template <typename Exp>
  Matrix(const _RValue<Scalar, Exp>& exp) {
    static_assert(Exp::ROWS_ == ROWS_ && Exp::COLS_ == COLS_,
                  "Cannot construct a matrix from a matrix of a differnt size");

    for (int i = 0; i < ROWS_; i++) {
      for (int j = 0; j < COLS_; j++) {
        (*this)(i, j) = exp(i, j);
      }
    }
  }

#ifdef FIBER_USE_VARIADIC_TEMPLATES
  template <int Index>
  void BulkAssign() {}

  template <int Index, typename... Tail>
  void BulkAssign(Scalar head, Tail... tail) {
    static_assert(Index < Rows * Cols, "Too many inputs to BulkAssign!");
    (*this)[Index] = head;
    BulkAssign<Index + 1>(tail...);
  }

  /// We require at least two values so as to disambiguate the constructor
  /// from the construct-by-rvalue constructor
  template <typename... Scalars>
  Matrix(Scalar a, Scalar b, Scalars... scalars) {
    BulkAssign<0>(a, b, scalars...);
  }
#else

  /// Fixed size construction
  Matrix(Scalar x0, Scalar x1, Scalar x2, Scalar x3) {
    static_assert(SIZE_ == 4,
                  "This constructor is only for size 4 matrices");
    data_[0] = x0;
    data_[1] = x1;
    data_[2] = x2;
    data_[3] = x3;
  }

  /// Fixed size construction
  Matrix(Scalar x0, Scalar x1, Scalar x2) {
    static_assert(SIZE_ == 3,
                  "This constructor is only for size 3 matrices");
    data_[0] = x0;
    data_[1] = x1;
    data_[2] = x2;
  }

  /// Fixed size construction
  Matrix(Scalar x0, Scalar x1) {
    static_assert(SIZE_ == 2,
                  "This constructor is only for size 2 matrices");
    data_[0] = x0;
    data_[1] = x1;
  }
#endif  // USE_VARIADIC_TEMPLATES

  Matrix(Scalar a) {
    static_assert(SIZE_ == 1,
                  "This constructor is only for size 1 matrices");
    data_[0] = a;
  }

};

typedef Matrix<double,2,2> Matrix2d;
typedef Matrix<double,3,3> Matrix3d;
typedef Matrix<double,4,4> Matrix4d;

typedef Matrix<double,2,1> Vector2d;
typedef Matrix<double,3,1> Vector3d;
typedef Matrix<double,4,1> Vector4d;

typedef Matrix<float,2,2> Matrix2f;
typedef Matrix<float,3,3> Matrix3f;
typedef Matrix<float,4,4> Matrix4f;

typedef Matrix<float,2,1> Vector2f;
typedef Matrix<float,3,1> Vector3f;
typedef Matrix<float,4,1> Vector4f;

}  // namespace fiber


#endif  // FIBER_MATRIX_H_
