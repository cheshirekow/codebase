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

#ifndef FIBER_NORMALIZE_H_
#define FIBER_NORMALIZE_H_

#include <cmath>


namespace fiber {

template <typename Scalar, class Exp>
inline Scalar Norm(_RValue<Scalar, Exp> const& A) {
  Scalar r(0);
  for (int i = 0; i < A.size(); i++) {
    r += A[i] * A[i];
  }
  return std::sqrt(r);
}

/// expression template for difference of two matrix expressions
template <typename Scalar, class Exp>
class _Normalize : public _RValue<Scalar, _Normalize<Scalar, Exp> > {
  Scalar norm_;
  Exp const& v_;

 public:
  typedef unsigned int Size_t;

  _Normalize(Exp const& A) : norm_(Norm(A)), v_(A) {}

  Size_t size() const { return v_.size(); }
  Size_t rows() const { return v_.rows(); }
  Size_t cols() const { return v_.cols(); }

  Scalar operator[](Size_t i) const {
    return v_[i] / norm_;
  }

  Scalar operator()(Size_t i, Size_t j) const {
    return v_(i, j) / norm_;
  }
};

template <typename Scalar, class Exp>
inline _Normalize<Scalar, Exp> Normalize(_RValue<Scalar, Exp> const& A) {
  return _Normalize<Scalar, Exp>(static_cast<Exp const&>(A));
}

}  // namespace fiber


#endif  // FIBER_NORMALIZE_H_
