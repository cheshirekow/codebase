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
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef MPBLOCKS_LINALG_STREAMASSIGNMENT_H_
#define MPBLOCKS_LINALG_STREAMASSIGNMENT_H_

namespace mpblocks {
namespace linalg {

/// assignment
template <class Mat>
class StreamAssignment {
 public:
  typedef StreamAssignment<Mat> Stream_t;

 private:
  Mat& m_mat;
  unsigned int m_i;

 public:
  StreamAssignment(Mat& M) : m_mat(M), m_i(0) {}

  template <typename Format_t>

  Stream_t& operator<<(Format_t x) {
    m_mat[m_i++] = x;
    return *this;
  }

  template <typename Format_t>

  Stream_t& operator, (Format_t x) {
    m_mat[m_i++] = x;
    return *this;
  }
};

}  // namespace linalg
}  // namespace mpblocks

#endif  // MPBLOCKS_LINALG_STREAMASSIGNMENT_H_
