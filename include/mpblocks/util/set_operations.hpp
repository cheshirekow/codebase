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
 *
 *  @date   Sept 17, 2014
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */
#ifndef MPBLOCKS_UTIL_SET_OPERATIONS_HPP_
#define MPBLOCKS_UTIL_SET_OPERATIONS_HPP_

#include <algorithm>

namespace set {

/// Returns true if the seconds set is a subset of the first set
template <typename InputIt1, typename InputIt2>
bool Contains(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

/// Collect in out1 elements from the first set not found in the second, and
/// collect in out2 elements from the second set not found in the first
template <typename InputIt1, typename InputIt2, typename OutputIt1,
          typename OutputIt2>
void SymmetricDifference(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                         InputIt2 last2, OutputIt1 out1, OutputIt2 out2);

/// Collect in out1 elements from the first set not found in the second, and
/// collect in out2 elements from the second set not found in the first, and
/// collect in intersect the elements from first and second that are common
/// to both.
template <typename InputIt1, typename InputIt2, typename OutputIt1,
          typename OutputIt2, typename OutputIt3>
void IntersectionAndDifference(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                               InputIt2 last2, OutputIt1 out1, OutputIt2 out2,
                               OutputIt3 intersect);

}  // namespace std


// ----------------------------------------------------------------------------
//                          Implementation
// ----------------------------------------------------------------------------

namespace set {

/// Returns true if the seconds set is a subset of the first set
template <typename InputIt1, typename InputIt2>
bool Contains(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                  InputIt2 last2) {
  while (first2 != last2) {
    for (; *first1 < *first2; ++first1) {
      if (first1 == last1) {
        return false;
      }
    }
    if (*first2 < *first1) {
      return false;
    } else {
      ++first2;
      ++first1;
    }
  }
  return true;
}

template <typename InputIt1, typename InputIt2, typename OutputIt1,
          typename OutputIt2>
void SymmetricDifference(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                         InputIt2 last2, OutputIt1 out1, OutputIt2 out2) {
  while (first1 != last1) {
    if (first2 == last2) {
      std::copy(first1, last1, out1);
      return;
    }
    if (*first1 < *first2) {
      *out1++ = *first1++;
    } else {
      if (*first2 < *first1) {
        *out2++ = *first2;
      } else {
        ++first1;
      }
      ++first2;
    }
  }
  std::copy(first2, last2, out2);
  return;
}

template <typename InputIt1, typename InputIt2, typename OutputIt1,
          typename OutputIt2, typename OutputIt3>
void IntersectionAndDifference(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                               InputIt2 last2, OutputIt1 out1, OutputIt2 out2,
                               OutputIt3 intersect) {
  while (first1 != last1) {
    if (first2 == last2) {
      std::copy(first1, last1, out1);
      return;
    }
    // *first1 is in the first set, but not the second set
    if (*first1 < *first2) {
      *out1++ = *first1++;
    } else {
      // *first2 is in the second set, but not the first
      if (*first2 < *first1) {
        *out2++ = *first2++;
        // *first1 == *first2 and it is in both sets
      } else {
        *intersect++ = *first1++;
        ++first2;
      }
    }
  }
  std::copy(first2, last2, out2);
  return;
}

}  // namespace std

#endif  // MPBLOCKS_UTIL_SET_OPERATIONS_HPP_
