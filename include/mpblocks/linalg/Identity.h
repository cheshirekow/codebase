/*
 *  Copyright (C) 2014 Josh Bialkowski (jbialk@mit.edu)
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
 *  @date   Dec 12, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef MPBLOCKS_LINALG_IDENTITY_H_
#define MPBLOCKS_LINALG_IDENTITY_H_

namespace mpblocks {
namespace linalg {

template <typename Scalar, int Size>
class Eye :  public _RValue<Scalar, Eye<Scalar, Size> > {
 public:
  int size() const { return Size*Size; }
  int rows() const { return Size; }
  int cols() const { return Size; }

  /// vector accessor
  Scalar operator[](int i) const {
    if(i % Size == i/Size) {
      return Scalar(1.0);
    } else {
      return Scalar(0.0);
    }
  }

  /// matrix accessor
  Scalar operator()(int i, int j) const {
    assert(i < Size && j < Size);
    if(i == j) {
      return Scalar(1.0);
    } else {
      return Scalar(0.0);
    }
  }
};

typedef Eye<double,2> Eye2d;
typedef Eye<double,3> Eye3d;
typedef Eye<double,4> Eye4d;

typedef Eye<float,2> Eye2f;
typedef Eye<float,3> Eye3f;
typedef Eye<float,4> Eye4f;

}  // namespace linalg
}  // namespace mpblocks

#endif  // MPBLOCKS_LINALG_IDENTITY_H_
