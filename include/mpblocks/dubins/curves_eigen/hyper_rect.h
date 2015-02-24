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
 *  @file   /home/josh/Codes/cpp/mpblocks2/dubins/include/mpblocks/dubins/curves_eigen/hyper.hpp
 *
 *  @date   Jun 26, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_H_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {
namespace        hyper {

enum ConstraintSpec
{
    OFF,
    MIN,
    MAX,
};

/// A hyper-rectangle in dubins space: A rectangular prism in R^3
template <typename Format_t>
struct HyperRect {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d;

  Vector3d minExt;
  Vector3d maxExt;

  /// Returns true if the dubins query state @p q lies within this region
  /// of dubins states.
  bool contains(const Vector3d& q) const {
    for (int i = 0; i < 3; i++) {
      if (q[i] > maxExt[i] || q[i] < minExt[i]) return false;
    }

    return true;
  }
};

} // namespace hyper
} // namespace curves_eigen
} // namespace dubins
} // namespace mpblocks

#include <mpblocks/dubins/curves_eigen/hyper/SpecPack.h>














#endif // HYPER_HPP_
