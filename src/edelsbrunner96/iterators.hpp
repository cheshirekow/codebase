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

#ifndef EDELSBRUNNER96_ITERATORS_HPP_
#define EDELSBRUNNER96_ITERATORS_HPP_

#include <edelsbrunner96/iterators.h>

namespace edelsbrunner96 {
namespace iter {

template <class Traits>
inline BreadthFirst<Traits>& BreadthFirst<Traits>::operator++() {
  if (queue_iter_ != queue_->end()) {
    SimplexRef simplex_ref = *queue_iter_;
    for (SimplexRef neighbor_ref : storage_[simplex_ref].N) {
      if (!storage_[neighbor_ref].marked[simplex::BFS_QUEUED]) {
        storage_[neighbor_ref].marked[simplex::BFS_QUEUED] = true;
        queue_->push_back(neighbor_ref);
      }
    }
  }
  ++queue_iter_;
  return *this;
}

}  // namespace iter
}  // namespace edelsbrunner96

#endif  // EDELSBRUNNER96_ITERATORS_HPP_
