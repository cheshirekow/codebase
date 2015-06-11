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
#ifndef CLARKSON93_P_QUEUE_H_
#define CLARKSON93_P_QUEUE_H_

#include <queue>

namespace clarkson93 {

/// A priority queue with a slightly more readable interface than from the STL
template <class T>
class P_Queue
    : public std::priority_queue<T, std::vector<T>, std::greater<T> > {
 public:
  typedef std::priority_queue<T, std::vector<T>, std::greater<T> > Base;
  typedef typename Base::size_type size_t;

  typename std::vector<T>::iterator begin() { return this->c.begin(); }

  typename std::vector<T>::iterator end() { return this->c.begin(); }

  void reserve(size_t cap) { this->c.reserve(cap); }

  void clear() { this->c.clear(); }

  T pop() {
    T tmp = this->Base::top();
    this->Base::pop();
    return tmp;
  }
};

}  // namespace clarkson93

#endif  // CLARKSON93_P_QUEUE_H_
