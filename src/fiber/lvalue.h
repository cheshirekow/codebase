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
template <typename Scalar, class Exp>
class _LValue {
 public:
  typedef StreamAssignment<_LValue<Scalar, Exp> > StreamAssign;

  Size size() const { return static_cast<Exp const&>(*this).size(); }
  Size rows() const { return static_cast<Exp const&>(*this).rows(); }
  Size cols() const { return static_cast<Exp const&>(*this).cols(); }

  Scalar& operator[](Size i) { return static_cast<Exp&>(*this)[i]; }

  Scalar const& operator[](Size i) const {
    return static_cast<Exp const&>(*this)[i];
  }

  Scalar& operator()(Size i, Size j) {
    return static_cast<Exp&>(*this)(i, j);
  }

  Scalar const& operator()(Size i, Size j) const {
    return static_cast<Exp const&>(*this)(i, j);
  }

  /// returns a stream for assignment
  StreamAssign operator<<(Scalar x) {
    StreamAssign stream(*this);
    return stream << x;
  }

  template <class Exp2>
  _LValue<Scalar, Exp>& operator=(_RValue<Scalar, Exp2> const& B) {
    assert(rows() == B.rows());
    assert(cols() == B.cols());
    for (int i = 0; i < rows(); i++) {
      for (int j = 0; j < cols(); j++) {
        (*this)(i, j) = B(i, j);
      }
    }
    return *this;
  }
};

/// Explicitly expose _LValue of an expressions, can be used to help the
/// compiler disambiguate overloads
template <typename Scalar, class Exp>
_LValue<Scalar, Exp>& LValue(_LValue<Scalar, Exp>& exp) {
  return exp;
}

}  // namespace fiber


#endif  // FIBER_LVALUE_H_
