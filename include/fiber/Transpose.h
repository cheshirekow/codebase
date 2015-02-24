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

#ifndef FIBER_TRANSPOSE_H_
#define FIBER_TRANSPOSE_H_


namespace fiber {

/// expression template for difference of two matrix expressions
template <typename Scalar, class Exp>
class _Transpose : public _RValue<Scalar, _Transpose<Scalar, Exp> > {
  Exp const& m_A;

 public:
  typedef unsigned int Size_t;

  _Transpose(Exp const& A) : m_A(A) {}

  Size_t size() const { return m_A.size(); }
  Size_t rows() const { return m_A.cols(); }
  Size_t cols() const { return m_A.rows(); }

  Scalar operator[](Size_t i) const { return m_A[i]; }
  Scalar operator()(Size_t i, Size_t j) const { return m_A(j, i); }
};

template <typename Scalar, class Exp>
inline _Transpose<Scalar, Exp> Transpose(_RValue<Scalar, Exp> const& A) {
  return _Transpose<Scalar, Exp>(static_cast<Exp const&>(A));
}

}  // namespace fiber


#endif  // FIBER_TRANSPOSE_H_
