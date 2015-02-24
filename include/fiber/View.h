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

#ifndef FIBER_VIEW_H_
#define FIBER_VIEW_H_


namespace fiber   {

/// expression template for subset of a matrix expression
template <typename Scalar, class Exp>
class _RView : public _RValue<Scalar, _RView<Scalar, Exp> > {
 public:
  typedef unsigned int Size_t;

 protected:
  Exp const& A_;
  Size_t i_;
  Size_t j_;
  Size_t rows_;
  Size_t cols_;

 public:
  _RView(Exp const& A, Size_t i, Size_t j, Size_t rows_in, Size_t cols_in)
      : A_(A), i_(i), j_(j), rows_(rows_in), cols_(cols_in) {}

  Size_t size() const { return rows_ * cols_; }
  Size_t rows() const { return rows_; }
  Size_t cols() const { return cols_; }

  Scalar operator[](Size_t k) const {
    Size_t i = k / cols_;
    Size_t j = k - i * cols_;
    return (*this)(i, j);
  }

  Scalar operator()(Size_t i, Size_t j) const {
    return A_(i_ + i, j_ + j);
  }
};

template <typename Scalar, class Exp>
inline _RView<Scalar, Exp> View(_RValue<Scalar, Exp> const& A, int i, int j,
                                int rows, int cols) {
  return _RView<Scalar, Exp>(static_cast<Exp const&>(A), i, j, rows, cols);
}

template <typename Scalar, class Exp>
inline _RView<Scalar, Exp> ViewRow(_RValue<Scalar, Exp> const& A, int i) {
  return _RView<Scalar, Exp>(static_cast<Exp const&>(A), i, 0, 1, A.cols());
}

template <typename Scalar, class Exp>
inline _RView<Scalar, Exp> ViewRows(_RValue<Scalar, Exp> const& A, int i,
                                    int nrows) {
  return _RView<Scalar, Exp>(static_cast<Exp const&>(A), i, 0, nrows, A.cols());
}

template <typename Scalar, class Exp>
inline _RView<Scalar, Exp> ViewColumn(_RValue<Scalar, Exp> const& A, int j) {
  return _RView<Scalar, Exp>(static_cast<Exp const&>(A), 0, j, A.rows(), 1);
}

template <typename Scalar, class Exp>
inline _RView<Scalar, Exp> ViewColumns(_RValue<Scalar, Exp> const& A, int j,
                                       int ncols) {
  return _RView<Scalar, Exp>(static_cast<Exp const&>(A), 0, j, A.rows(), ncols);
}

/// expression template for subset of a matrix expression
template <typename Scalar, class Exp>
class LView : public _LValue<Scalar, LView<Scalar, Exp> >,
              public _RValue<Scalar, LView<Scalar, Exp> > {
 public:
  typedef unsigned int Size_t;

 protected:
  Exp& A_;
  Size_t i_;
  Size_t j_;
  Size_t rows_;
  Size_t cols_;

 public:
  LView(Exp& A, Size_t i, Size_t j, Size_t rows_in, Size_t cols_in)
      : A_(A), i_(i), j_(j), rows_(rows_in), cols_(cols_in) {}

  Size_t size() const { return rows_ * cols_; }
  Size_t rows() const { return rows_; }
  Size_t cols() const { return cols_; }

  /// return the evaluated i'th element of a vector expression
  Scalar const& operator[](Size_t k) const {
    Size_t i = k / cols_;
    Size_t j = k - i * cols_;
    return (*this)(i, j);
  }

  /// return the evaluated i'th element of a vector expression
  Scalar& operator[](Size_t k) {
    Size_t i = k / cols_;
    Size_t j = k - i * cols_;
    return (*this)(i, j);
  }

  /// return the evaluated (j,i)'th element of a matrix expression
  Scalar const& operator()(Size_t i, Size_t j) const {
    return A_(i_ + i, j_ + j);
  }

  /// return the evaluated (j,i)'th element of a matrix expression
  Scalar& operator()(Size_t i, Size_t j) { return A_(i_ + i, j_ + j); }

  template <class Exp2>
  LView<Scalar, Exp>& operator=(_RValue<Scalar, Exp2> const& B) {
    _LValue<Scalar, LView<Scalar, Exp> >::operator=(B);
    return *this;
  }

  template <class Exp2>
  LView<Scalar, Exp>& operator=(_LValue<Scalar, Exp2> const& B) {
    _LValue<Scalar, LView<Scalar, Exp> >::operator=(B);
    return *this;
  }
};

template <typename Scalar, class Exp>
inline LView<Scalar, Exp> Block(_LValue<Scalar, Exp>& A, int i, int j, int rows,
                                int cols) {
  return LView<Scalar, Exp>(static_cast<Exp&>(A), i, j, rows, cols);
}

template <typename Scalar, class Exp>
inline LView<Scalar, Exp> Row(_LValue<Scalar, Exp>& A, int i) {
  return LView<Scalar, Exp>(static_cast<Exp&>(A), i, 0, 1, A.cols());
}

template <typename Scalar, class Exp>
inline LView<Scalar, Exp> Rows(_LValue<Scalar, Exp>& A, int i, int nrows) {
  return LView<Scalar, Exp>(static_cast<Exp&>(A), i, 0, nrows, A.cols());
}

template <typename Scalar, class Exp>
inline LView<Scalar, Exp> Column(_LValue<Scalar, Exp>& A, int j) {
  return LView<Scalar, Exp>(static_cast<Exp&>(A), 0, j, A.rows(), 1);
}

template <typename Scalar, class Exp>
inline LView<Scalar, Exp> Columns(_LValue<Scalar, Exp>& A, int j, int ncols) {
  return LView<Scalar, Exp>(static_cast<Exp&>(A), 0, j, A.rows(), ncols);
}

}  // namespace fiber


#endif  // FIBER_VIEW_H_
