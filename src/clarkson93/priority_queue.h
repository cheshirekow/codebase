/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
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
#ifndef CLARKSON93_PRIORITY_QUEUE_H_
#define CLARKSON93_PRIORITY_QUEUE_H_

#include <queue>

namespace clarkson93 {

/// A priority queue with a slightly more readable interface than from the STL
/**
 *  In particular we expose some of the functions of the underlying container,
 *  as well as provide a pop method that returns the popped value.
 */
template <class T>
class PriorityQueue
    : public std::priority_queue<T, std::vector<T>, std::greater<T> > {
 public:
  typedef std::priority_queue<T, std::vector<T>, std::greater<T> > Base;
  typedef typename Base::size_type size_type;

  /// return an iterator to the front of the underlying vector
  typename std::vector<T>::iterator begin() {
    return this->c.begin();
  }

  /// return an iterator to the past-the-end of the underlying vector
  typename std::vector<T>::iterator end() {
    return this->c.end();
  }

  /// reserve space in the underlying vector
  void Reserve(size_type cap) {
    this->c.reserve(cap);
  }

  /// clear the underlying vector
  void Clear() {
    this->c.clear();
  }

  /// remove the next element in the queue and return it
  T Pop() {
    T tmp = this->Base::top();
    this->Base::pop();
    return tmp;
  }
};

}  // namespace clarkson93

#endif  // CLARKSON93_PRIORITY_QUEUE_H_
