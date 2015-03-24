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
 *  @date   Oct 25, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef EDELSBRUNNER96_ITERATORS_H_
#define EDELSBRUNNER96_ITERATORS_H_

#include <list>
#include <edelsbrunner96/simplex.h>

namespace edelsbrunner96 {

namespace iter {

template <class Traits>
class BreadthFirst {
 public:
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::Storage Storage;

  BreadthFirst(Storage& storage, std::list<SimplexRef>* queue,
               typename std::list<SimplexRef>::iterator queue_iter)
      : storage_(storage), queue_(queue), queue_iter_(queue_iter) {}

  bool operator!=(const BreadthFirst<Traits>& other) {
    return queue_iter_ != other.queue_iter_;
  }

  SimplexRef operator*() {
    return *queue_iter_;
  }

  BreadthFirst<Traits>& operator++();

 private:
  Storage& storage_;
  std::list<SimplexRef>* queue_;
  typename std::list<SimplexRef>::iterator queue_iter_;
};

}  // namespace iter

template <class Traits>
class BreadthFirst {
 public:
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::Storage Storage;

  BreadthFirst(Storage& storage, SimplexRef start_ref)
      : storage_(storage), start_ref_(start_ref) {
    storage_[start_ref].marked[simplex::BFS_QUEUED] = true;
    queue_.push_back(start_ref);
  }

  ~BreadthFirst() {
    for (auto s_ref : queue_) {
      storage_[s_ref].marked[simplex::BFS_QUEUED] = false;
    }
  }

  iter::BreadthFirst<Traits> begin() {
    return iter::BreadthFirst<Traits>(storage_, &queue_, queue_.begin());
  }

  iter::BreadthFirst<Traits> end() {
    return iter::BreadthFirst<Traits>(storage_, &queue_, queue_.end());
  }

 private:
  Storage& storage_;
  SimplexRef start_ref_;
  std::list<SimplexRef> queue_;
};

}  // namespace edelsbrunner96

#endif  // EDELSBRUNNER96_ITERATORS_H_
