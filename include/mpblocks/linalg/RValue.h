/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of mpblocks.
 *
 *  mpblocks is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  mpblocks is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with mpblocks.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef MPBLOCKS_LINALG_RVALUE_H_
#define MPBLOCKS_LINALG_RVALUE_H_

namespace mpblocks {
namespace linalg {

/// expression template for rvalues
template <typename Scalar, class Mat>
class _RValue {
 public:
  typedef unsigned int Size_t;

  Size_t size() const { return static_cast<Mat const&>(*this).size(); }
  Size_t rows() const { return static_cast<Mat const&>(*this).rows(); }
  Size_t cols() const { return static_cast<Mat const&>(*this).cols(); }

  Scalar operator[](Size_t i) const {
    return static_cast<Mat const&>(*this)[i];
  }

  Scalar operator()(Size_t i, Size_t j) const {
    return static_cast<Mat const&>(*this)(i, j);
  }

  operator Mat&() { return static_cast<Mat&>(*this); }
  operator Mat const&() { return static_cast<Mat const&>(*this); }
};

template <typename Scalar, class Mat>
const _RValue<Scalar, Mat>& RValue(const _RValue<Scalar, Mat>& exp) {
  return exp;
}

}  // namespace linalg
}  // namespace mpblocks

#endif  // MPBLOCKS_LINALG_RVALUE_H_
