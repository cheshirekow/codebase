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
 *  @date   Sep 14, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  shamelessly borred from
 *          http://akrzemi1.wordpress.com/2012/10/23/user-defined-literals-part-ii/
 */

#ifndef MPBLOCKS_UTIL_BINARY_LITERAL_H_
#define MPBLOCKS_UTIL_BINARY_LITERAL_H_

#include <limits>
#include <stdexcept>

namespace mpblocks {
namespace  utility {

template <typename T>
constexpr size_t NumberOfBits() {
  static_assert(std::numeric_limits<T>::is_integer, "only integers allowed");

  // from en.cppreference.com: The value of std::numeric_limits<T>::digits is
  // the number of digits in base-radix that can be represented by the type T
  // without change. For integer types, this is the number of bits not counting
  // the sign bit. For floating-point types, this is the number of digits in
  // the mantissa.
  return std::numeric_limits<T>::digits;
}

// compute the length of a string using recursion, so that it can be
// compile-time evaluated
constexpr size_t StringLength(const char* str, size_t current_len = 0) {
  return *str == '\0'
             ? current_len                              // end of recursion
             : StringLength(str + 1, current_len + 1);  // compute recursively
}

// validates that a character is part of a binary string
constexpr bool IsBinary(char c) { return c == '0' || c == '1'; }

// implementation function called after validating string length
template <typename OutputType=unsigned>
constexpr unsigned _BinaryLiteral(const char* str, size_t val = 0) {
  return StringLength(str) == 0
             ? val             // end of recursion
             : IsBinary(*str)  // check for non-binary digit
                   ? _BinaryLiteral(str + 1, 2 * val + *str - '0')
                   : throw std::logic_error("char is not '0' or '1'");
}

template <typename OutputType=unsigned>
constexpr OutputType BinaryLiteral(const char* str, size_t val = 0) {
  return StringLength(str) <= NumberOfBits<OutputType>()
             ? _BinaryLiteral(str,val)
             : throw std::logic_error("Binary literal is too long for type");
}

}  // namespace utility
}  // namespace mpblocks

#define BINARY_LITERAL(X) mpblocks::utility::BinaryLiteral(#X)

#if (__cplusplus >= 201103L)
constexpr unsigned operator"" _b( const char * str ) {
  return mpblocks::utility::BinaryLiteral(str);
}
#endif

#endif // MPBLOCKS_UTIL_BINARY_LITERAL_H_
