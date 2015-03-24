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
template<typename Scalar, class Exp, Size rows_, Size cols_>
class _RView : public _RValue<Scalar, _RView<Scalar, Exp, rows_, cols_> > {
 protected:
  Exp const& A_;   ///< underlying matrix expression
  Index i_;
  Index j_;

 public:
  enum {
    ROWS_ = rows_,
    COLS_ = cols_,
    SIZE_ = rows_ * cols_
  };

  _RView(Exp const& A, int i, int j)
      : A_(A), i_(i), j_(j) {
    assert(i + ROWS_ <= Exp::ROWS_
          && j + COLS_ <= Exp::COLS_);
  }

  Size size() const { return rows_ * cols_; }
  Size rows() const { return rows_; }
  Size cols() const { return cols_; }

  Scalar operator[](Index k) const {
    assert(k < SIZE_);
    Index i = k / cols_;
    Index j = k - i * cols_;
    return (*this)(i, j);
  }

  Scalar operator()(Index i, Index j) const {
    assert(i < rows_ && j < cols_);
    return A_(i_ + i, j_ + j);
  }
};

template<int rows, int cols, typename Scalar, class Exp>
inline _RView<Scalar, Exp, rows, cols> View(_RValue<Scalar, Exp> const& A,
    Index i, Index j) {
  return _RView<Scalar, Exp, rows, cols>(static_cast<Exp const&>(A), i, j);
}

template<typename Scalar, class Exp>
inline _RView<Scalar, Exp, 1, Exp::COLS_> GetRow(_RValue<Scalar, Exp> const& A,
    int i) {
  return _RView<Scalar, Exp, 1, Exp::COLS_>(static_cast<Exp const&>(A), i, 0);
}

template<int rows, typename Scalar, class Exp>
inline _RView<Scalar, Exp, rows, Exp::COLS_> GetRows(
    _RValue<Scalar, Exp> const& A, int i) {
  return _RView<Scalar, Exp, rows, Exp::COLS_>(static_cast<Exp const&>(A),
                                               i, 0);
}

template<typename Scalar, class Exp>
inline _RView<Scalar, Exp, Exp::ROWS_, 1> GetColumn(
    _RValue<Scalar, Exp> const& A, int j) {
  return _RView<Scalar, Exp, Exp::ROWS_, 1>(static_cast<Exp const&>(A), 0, j);
}

template< int cols, typename Scalar, class Exp>
inline _RView<Scalar, Exp, Exp::ROWS_, cols> GetColumns(
    _RValue<Scalar, Exp> const& A, int j) {
  return _RView<Scalar, Exp, Exp::ROWS_, cols>(static_cast<Exp const&>(A),
                                               0, j);
}

/// expression template for subset of a matrix expression
template <typename Scalar, class Exp, int rows_, int cols_>
class LView : public _LValue<Scalar, LView<Scalar, Exp, rows_, cols_> > {

 protected:
  Exp& A_;
  Index i_;
  Index j_;

 public:
  typedef LView<Scalar, Exp, rows_, cols_> ThisType;
  typedef _LValue<Scalar, ThisType> LValueType;

  enum {
    ROWS_ = rows_,
    COLS_ = cols_,
    SIZE_ = rows_ * cols_
  };

  LView(Exp& A, Index i, Index j)
      : A_(A), i_(i), j_(j) {}

  Size size() const { return rows_ * cols_; }
  Size rows() const { return rows_; }
  Size cols() const { return cols_; }

  /// return the evaluated i'th element of a vector expression
  Scalar const& operator[](Index k) const {
    assert(k < SIZE_);
    Index i = k / cols_;
    Index j = k - i * cols_;
    return (*this)(i, j);
  }

  /// return the evaluated i'th element of a vector expression
  Scalar& operator[](Index k) {
    assert(k < SIZE_);
    Index i = k / cols_;
    Index j = k - i * cols_;
    return (*this)(i, j);
  }

  /// return the evaluated (j,i)'th element of a matrix expression
  Scalar const& operator()(Index i, Index j) const {
    return A_(i_ + i, j_ + j);
  }

  /// return the evaluated (j,i)'th element of a matrix expression
  Scalar& operator()(Index i, Index j) {
    return A_(i_ + i, j_ + j);
  }

  template <class OtherExp>
  ThisType& operator=(const _RValue<Scalar, OtherExp>& other) {
    LValueType::operator=(other);
    return *this;
  }
};

template<int rows, int cols, typename Scalar, class Exp>
inline LView<Scalar, Exp, rows, cols> Block(_LValue<Scalar, Exp>& A, int i,
    int j) {
  return LView<Scalar, Exp, rows, cols>(static_cast<Exp&>(A), i, j);
}

template<typename Scalar, class Exp>
inline LView<Scalar, Exp, 1, Exp::COLS_> Row(_LValue<Scalar, Exp>& A, int i) {
  return LView<Scalar, Exp, 1, Exp::COLS_>(static_cast<Exp&>(A), i, 0);
}

template<int rows, typename Scalar, class Exp>
inline LView<Scalar, Exp, rows, Exp::COLS_> Rows(_LValue<Scalar, Exp>& A,
    int i) {
  return LView<Scalar, Exp, rows, Exp::COLS_>(static_cast<Exp&>(A), i, 0);
}

template<typename Scalar, class Exp>
inline LView<Scalar, Exp, Exp::ROWS_, 1> Column(_LValue<Scalar, Exp>& A,
    int j) {
  return LView<Scalar, Exp, Exp::ROWS_, 1>(static_cast<Exp&>(A), 0, j);
}

template<int cols, typename Scalar, class Exp>
inline LView<Scalar, Exp, Exp::ROWS_, cols> Columns(_LValue<Scalar, Exp>& A,
    int j) {
  return LView<Scalar, Exp, Exp::ROWS_, cols>(static_cast<Exp&>(A), 0, j);
}

}  // namespace fiber

#endif  // FIBER_VIEW_H_
