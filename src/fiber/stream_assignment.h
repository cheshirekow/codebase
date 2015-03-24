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
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef FIBER_STREAMASSIGNMENT_H_
#define FIBER_STREAMASSIGNMENT_H_


namespace fiber {

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

}  // namespace fiber


#endif  // FIBER_STREAMASSIGNMENT_H_
