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

#ifndef FIBER_PRODUCT_H_
#define FIBER_PRODUCT_H_


namespace fiber   {

/// expression template for product of two matrix expressions
template <typename Scalar, class Exp1, class Exp2>
class Product : public _RValue<Scalar, Product<Scalar, Exp1, Exp2> > {
  Exp1 const& A_;
  Exp2 const& B_;

 public:
  typedef unsigned int Size_t;

  Product(Exp1 const& A, Exp2 const& B) : A_(A), B_(B) {}

  /// return the size for a vector
  Size_t size() const { return (A_.rows() * B_.cols()); }

  /// return the rows of a matrix expression
  Size_t rows() const { return A_.rows(); }

  /// return the columns of a matrix expression
  Size_t cols() const { return B_.cols(); }

  /// return the evaluated i'th element of a vector expression
  Scalar operator[](Size_t i) const { return (*this)(i / cols(), i % cols()); }

  /// return the evaluated (i,j)'th element of a matrix expression
  Scalar operator()(Size_t i, Size_t j) const {
    Scalar r(0);
    for (int k = 0; k < A_.cols(); k++) {
      r += A_(i, k) * B_(k, j);
    }
    return r;
  }
};

template <typename Scalar, class Exp1, class Exp2>
inline Product<Scalar, Exp1, Exp2> operator*(_RValue<Scalar, Exp1> const& A,
                                             _RValue<Scalar, Exp2> const& B) {
  typedef Product<Scalar, Exp1, Exp2> Product_t;
  return Product_t(static_cast<Exp1 const&>(A), static_cast<Exp2 const&>(B));
}

template <typename Scalar, class Exp1, class Exp2>
inline Scalar Dot(_RValue<Scalar, Exp1> const& A,
                  _RValue<Scalar, Exp2> const& B) {
  assert(A.size() == B.size());
  Scalar r(0);
  for (int i = 0; i < A.size(); i++) {
    r += A[i] * B[i];
  }
  return r;
}

}  // namespace fiber


#endif  // FIBER_PRODUCT_H_
