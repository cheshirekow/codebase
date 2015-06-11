/*
 *  Copyright (C) 2013 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of clarkson93.
 *
 *  clarkson93 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  clarkson93 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with clarkson93.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef CLARKSON93_INDEXED_H_
#define CLARKSON93_INDEXED_H_

namespace clarkson93 {

/// priority queue node, Lexicographical ordering by index and then value
/// TODO(josh): replace usages with std::tuple
template <typename Index, typename Value>
struct Indexed {
  Index idx;
  Value val;

  typedef Indexed<Index, Value> This;

  static bool LessThan(const This& a, const This& b) {
    return a.idx == b.idx ? a.val < b.val : a.idx < b.idx;
  }

  static bool GreaterThan(const This& a, const This& b) {
    return a.idx == b.idx ? a.val > b.val : a.idx > b.idx;
  }

  struct Less {
    bool operator()(const This& a, const This& b) {
      return This::LessThan(a, b);
    }
  };

  struct Greater {
    bool operator()(const This& a, const This& b) {
      return This::GreaterThan(a, b);
    }
  };
};

template <typename Index, typename Value>
inline bool operator<(const Indexed<Index, Value>& a,
                      const Indexed<Index, Value>& b) {
  return Indexed<Index, Value>::LessThan(a, b);
}

template <typename Index, typename Value>
inline bool operator>(const Indexed<Index, Value>& a,
                      const Indexed<Index, Value>& b) {
  return Indexed<Index, Value>::GreaterThan(a, b);
}

}  // namespace clarkson93

#endif  // CLARKSON93_INDEXED_H_
