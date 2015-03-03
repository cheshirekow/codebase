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

#ifndef FIBER_SCALE_H_
#define FIBER_SCALE_H_


namespace fiber   {

template <typename Scalar, class Exp>
class _Scale : public _RValue<Scalar, _Scale<Scalar, Exp> > {
  Scalar s_;
  Exp const& M_;

 public:
  enum {
    ROWS_ = Exp::ROWS_,
    COLS_ = Exp::COLS_,
    SIZE_ = Exp::SIZE_
  };

  _Scale(Scalar s, Exp const& A) : s_(s), M_(A) {}

  Size size() const { return M_.size(); }
  Size rows() const { return M_.rows(); }
  Size cols() const { return M_.cols(); }

  Scalar operator[](Size i) const { return s_ * M_[i]; }
  Scalar operator()(Size i, Size j) const { return s_ * M_(i, j); }
};

template <typename Scalar, class Exp>
inline _Scale<Scalar, Exp> operator*(Scalar s, _RValue<Scalar, Exp> const& A) {
  return _Scale<Scalar, Exp>(s, static_cast<Exp const&>(A));
}

template <typename Scalar, class Exp>
inline _Scale<Scalar, Exp> operator*(_RValue<Scalar, Exp> const& A, Scalar s) {
  return _Scale<Scalar, Exp>(s, static_cast<Exp const&>(A));
}

}  // namespace fiber


#endif  // FIBER_SCALE_H_
