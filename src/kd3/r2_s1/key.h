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
 *  @file   mpblocks/kd_tree/r2_s1/Key.h
 *
 *  @date   Nov 21, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_R2_S1_KEY_H_
#define MPBLOCKS_KD_TREE_R2_S1_KEY_H_

namespace mpblocks {
namespace kd_tree {
namespace r2_s1 {

template <class Traits>
struct Key {
  typedef typename Traits::Format_t Format_t;
  typedef typename Traits::Node Node_t;
  typedef Key<Traits> Key_t;

  Format_t d2;
  Node_t* n;

  struct Compare {
    bool operator()(const Key_t& a, const Key_t& b) {
      if (a.d2 < b.d2) return true;
      if (b.d2 < a.d2) return false;
      return (a.n < b.n);
    }
  };
};

}  // namespace r2_s1
}  // namespace kd_tree
}  // namespace mpblocks

#endif  // SEARCHKEY_H_
