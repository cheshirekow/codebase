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
 *  @date   Aug 4, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */
#ifndef MPBLOCKS_UTIL_RANGE_HPP_
#define MPBLOCKS_UTIL_RANGE_HPP_

namespace mpblocks {
namespace     util {

template <typename T>
struct Range_ {
  struct Iterator {
    T val;  ///< storage for the actual value

    Iterator(T val) : val(val) {}
    T operator*() { return val; }
    bool operator!=(T other) { return val != other; }
    Iterator& operator++() {
      ++val;
      return *this;
    }
    operator T() { return val; }
  };

 private:
  T m_begin;  ///< the first integral value
  T m_end;    ///< one past the last integral value

 public:
  Range_(T begin, T end) : m_begin(begin), m_end(end) {}

  T size() { return m_end - m_begin; }

  Iterator begin() { return m_begin; }
  Iterator end() { return m_end; }
};

template <typename T>
Range_<T> Range(T begin, T end) {
  return Range_<T>(begin, end);
}


}  // namespace util
}  // namespace mpblocks

#endif  // MPBLOCKS_UTIL_RANGE_HPP_
