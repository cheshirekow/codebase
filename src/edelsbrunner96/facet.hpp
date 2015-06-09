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

#ifndef EDELSBRUNNER96_FACET_HPP_
#define EDELSBRUNNER96_FACET_HPP_

#include <algorithm>
#include <Eigen/Dense>
#include <edelsbrunner96/facet.h>
#include <edelsbrunner96/simplex.h>

namespace edelsbrunner96 {

template<class Traits>
void Facet<Traits>::Construct(Storage& storage, SimplexRef s_0,
                              SimplexRef s_1) {
  s[0] = s_0;
  s[1] = s_1;
  for (int i = 0; i < 2; i++) {
    v[i] = storage[s[i]].version;
  }
}

template<class Traits>
bool Facet<Traits>::StillExists(Storage& storage) {
  for (int i = 0; i < 2; i++) {
    if (storage[s[i]].version != v[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace edelsbrunner

#endif  // EDELSBRUNNER96_FACET_HPP_
