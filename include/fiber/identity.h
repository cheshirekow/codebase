/*
 *  Copyright (C) 2014 Josh Bialkowski (jbialk@mit.edu)
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
 *  @date   Dec 12, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef FIBER_IDENTITY_H_
#define FIBER_IDENTITY_H_

namespace fiber {

/// A `N x N` matrix expression for the identity matrix
template <typename Scalar, int N>
class Eye :  public _RValue<Scalar, Eye<Scalar, N> > {
 public:
  enum {
    ROWS_ = N,
    COLS_ = N,
    SIZE_ = N * N
  };

  Size size() const { return N*N; }
  Size rows() const { return N; }
  Size cols() const { return N; }

  /// vector accessor
  Scalar operator[](Index i) const {
    if(i % N == i/N) {
      return Scalar(1.0);
    } else {
      return Scalar(0.0);
    }
  }

  /// matrix accessor
  Scalar operator()(int i, int j) const {
    assert(i < N && j < N);
    return (i == j) ? Scalar(1.0) : Scalar(0.0);
  }
};

typedef Eye<double,2> Eye2d;
typedef Eye<double,3> Eye3d;
typedef Eye<double,4> Eye4d;

typedef Eye<float,2> Eye2f;
typedef Eye<float,3> Eye3f;
typedef Eye<float,4> Eye4f;

}  // namespace fiber

#endif  // FIBER_IDENTITY_H_
