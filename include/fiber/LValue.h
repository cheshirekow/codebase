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

#ifndef FIBER_LVALUE_H_
#define FIBER_LVALUE_H_


namespace fiber {

/// expression template for rvalues
template <typename Scalar, class Mat>
class _LValue {
 public:
  typedef StreamAssignment<_LValue<Scalar, Mat> > Stream_t;

  typedef unsigned int Size_t;

  Size_t size() const { return static_cast<Mat const&>(*this).size(); }
  Size_t rows() const { return static_cast<Mat const&>(*this).rows(); }
  Size_t cols() const { return static_cast<Mat const&>(*this).cols(); }

  Scalar& operator[](Size_t i) { return static_cast<Mat&>(*this)[i]; }

  Scalar const& operator[](Size_t i) const {
    return static_cast<Mat const&>(*this)[i];
  }

  Scalar& operator()(Size_t i, Size_t j) {
    return static_cast<Mat&>(*this)(i, j);
  }

  Scalar const& operator()(Size_t i, Size_t j) const {
    return static_cast<Mat const&>(*this)(i, j);
  }

  /// returns a stream for assignment
  Stream_t operator<<(Scalar x) {
    Stream_t stream(*this);
    return stream << x;
  }

  template <class Exp2>
  _LValue<Scalar, Mat>& operator=(_RValue<Scalar, Exp2> const& B) {
    assert(rows() == B.rows());
    assert(cols() == B.cols());
    for (int i = 0; i < rows(); i++) {
      for (int j = 0; j < cols(); j++) {
        (*this)(i, j) = B(i, j);
      }
    }
    return *this;
  }

  template <class Exp2>
  _LValue<Scalar, Mat>& operator=(_LValue<Scalar, Exp2> const& B) {
    assert(rows() == B.rows());
    assert(cols() == B.cols());
    for (int i = 0; i < rows(); i++) {
      for (int j = 0; j < cols(); j++) {
        (*this)(i, j) = B(i, j);
      }
    }
    return *this;
  }

  operator _RValue<Scalar, _LValue<Scalar, Mat> >() {
    return _RValue<Scalar, _LValue<Scalar, Mat> >(*this);
  }
};

template <typename Scalar, class Mat>
_LValue<Scalar, Mat>& LValue(_LValue<Scalar, Mat>& exp) {
  return exp;
}

}  // namespace fiber


#endif  // FIBER_LVALUE_H_
