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

#ifndef MPBLOCKS_LINALG_OSTREAM_H_
#define MPBLOCKS_LINALG_OSTREAM_H_

#include <iostream>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace mpblocks {
namespace linalg {

template <typename Scalar, class Mat>
std::ostream& operator<<(std::ostream& out, _RValue<Scalar, Mat> const& M) {
  std::stringstream strm;
  int max_len = 0;
  for (int i = 0; i < M.rows(); i++) {
    for (int j = 0; j < M.cols(); j++) {
      strm.str("");
      strm << M(i, j);
      if (max_len < strm.str().size()) max_len = strm.str().size();
    }
  }

  out << std::setiosflags(std::ios::left);
  for (int i = 0; i < M.rows(); i++) {
    for (int j = 0; j < M.cols(); j++) {
      strm.str("");
      strm << M(i, j);
      out << std::setw(max_len) << strm.str() << "   ";
    }
    if (i < M.rows() - 1) {
      out << "\n";
    }
  }

  return out;
}

}  // namespace linalg
}  // namespace mpblocks

#endif  // MPBLOCKS_LINALG_OSTREAM_H_
