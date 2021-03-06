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

#ifndef FIBER_SUM_H_
#define FIBER_SUM_H_


namespace fiber {

/// expression template for sum of two matrix expressions
template <typename Scalar, class Exp1, class Exp2>
class _Sum : public _RValue<Scalar, _Sum<Scalar, Exp1, Exp2> > {
  Exp1 const& A_;
  Exp2 const& B_;

 public:
  enum {
    ROWS_ = Exp1::ROWS_,
    COLS_ = Exp1::COLS_,
    SIZE_ = Exp1::SIZE_
  };

  _Sum(Exp1 const& A, Exp2 const& B) : A_(A), B_(B) {
    static_assert(Exp1::ROWS_ == Exp2::ROWS_
                  && Exp1::COLS_ == Exp2::COLS_
                  && Exp1::SIZE_ == Exp2::SIZE_,
                  "Cannot sum matrices of different sizes");
  }

  Size size() const { return A_.size(); }
  Size rows() const { return A_.rows(); }
  Size cols() const { return A_.cols(); }

  Scalar operator[](Index i) const {
    return (A_[i] + B_[i]);
  }

  Scalar operator()(Index i, Index j) const {
    return (A_(i, j) + B_(i, j));
  }
};

template <typename Scalar, class Exp1, class Exp2>
inline _Sum<Scalar, Exp1, Exp2> operator+(_RValue<Scalar, Exp1> const& A,
                                          _RValue<Scalar, Exp2> const& B) {
  return _Sum<Scalar, Exp1, Exp2>(static_cast<Exp1 const&>(A),
                                  static_cast<Exp2 const&>(B));
}

}  // namespace fiber


#endif  // FIBER_SUM_H_SUM_H_
