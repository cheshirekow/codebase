/*
 *  Copyright (C) 2015 Josh Bialkowski (josh.bialkowski@gmail.com)
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
#ifndef CLARKSON93_ZIP_H_
#define CLARKSON93_ZIP_H_

#include <cassert>
#include <tuple>

namespace clarkson93 {

template <class IteratorA, class IteratorB>
struct ZipIterator {
  IteratorA iter_a;
  IteratorB iter_b;

  bool operator==(const ZipIterator<IteratorA, IteratorB>& other) {
    return (iter_a == other.iter_a) && (iter_b == other.iter_b);
  }

  bool operator!=(const ZipIterator<IteratorA, IteratorB>& other) {
    return !(*this == other);
  }

  auto operator*() -> decltype(std::make_tuple(*iter_a, *iter_b)) {
    return std::make_tuple(*iter_a, *iter_b);
  }

  const ZipIterator<IteratorA, IteratorB> operator++(int) {
    return ZipIterator<IteratorA, IteratorB>{iter_a++, iter_b++};
  }

  ZipIterator<IteratorA, IteratorB>& operator++() {
    ++iter_a;
    ++iter_b;
    return *this;
  }
};

template <class ContainerA, class ContainerB>
class ZipRange {
 public:
  typedef ZipIterator<typename ContainerA::iterator,
                      typename ContainerB::iterator> iterator;
  typedef ZipIterator<typename ContainerA::const_iterator,
                      typename ContainerB::const_iterator> const_iterator;

  ZipRange(ContainerA& a, ContainerB& b) : a_(a), b_(b) {}

  iterator begin() {
    return iterator{a_.begin(), b_.begin()};
  }

  const_iterator begin() const {
    return const_iterator{a_.c_begin(), b_.c_begin()};
  }

  const_iterator c_begin() const {
    return const_iterator{a_.c_begin(), b_.c_begin()};
  }

  iterator end() {
    return iterator{a_.end(), b_.end()};
  }

  const_iterator end() const {
    return const_iterator{a_.c_end(), b_.c_end()};
  }

  const_iterator c_end() const {
    return const_iterator{a_.c_end(), b_.c_end()};
  }

 private:
  ContainerA& a_;
  ContainerB& b_;
};

template <class ContainerA, class ContainerB>
ZipRange<ContainerA, ContainerB> Zip(const ContainerA& a, const ContainerB& b) {
  assert(a.size() == b.size());
  return ZipRange<ContainerA, ContainerB>{a, b};
}

template <class ContainerA, class ContainerB>
ZipRange<ContainerA, ContainerB> Zip(ContainerA& a, ContainerB& b) {
  assert(a.size() == b.size());
  return ZipRange<ContainerA, ContainerB>{a, b};
}

}  // namespace clarkson93

#endif  // CLARKSON93_ZIP_H_
