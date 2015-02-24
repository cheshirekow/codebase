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
 *  @file   /home/josh/Codes/cpp/mpblocks2/dubins/include/mpblocks/dubins/curves_eigen/hyper/Integrator.h
 *
 *  @date   Jun 30, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_INTEGRATOR_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_INTEGRATOR_H_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {

/// @see Integrate::solve
/**
 *  @tparam Scalar  number Scalar (double, float, etc)
 */
template <typename Scalar>
struct Integrate {
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3d;
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2d;
  typedef Path<Scalar> Path_t;

  static Vector3d L(const Vector3d& q0, const Scalar r, const Scalar arc) {
    Vector2d c = leftCenter(q0, r);
    Scalar theta1 = clampRadian(q0[2] + arc);
    Scalar alpha1 = leftAngleOf(theta1);

    return Vector3d(c[0] + r * std::cos(alpha1),
                    c[1] + r * std::sin(alpha1),
                    theta1);
  }

  static Vector3d R(const Vector3d& q0, const Scalar r, const Scalar arc) {
    Vector2d c = rightCenter(q0, r);
    Scalar theta1 = clampRadian(q0[2] - arc);
    Scalar alpha1 = rightAngleOf(theta1);

    return Vector3d(c[0] + r * std::cos(alpha1),
                    c[1] + r * std::sin(alpha1),
                    theta1);
  }

  static Vector3d S(const Vector3d& q0, const Scalar d1) {
    return Vector3d(q0[0] + d1 * std::cos(q0[2]),
                    q0[1] + d1 * std::sin(q0[2]),
                    q0[2]);
  }

  static Vector3d LRL(const Vector3d& q0, const Vector3d& s, const Scalar r) {
    Vector3d q1 = L(q0, r, s[0]);
    Vector3d q2 = R(q1, r, s[1]);
    return L(q2, r, s[2]);
  }

  static Vector3d RLR(const Vector3d& q0, const Vector3d& s, const Scalar r) {
    Vector3d q1 = R(q0, r, s[0]);
    Vector3d q2 = L(q1, r, s[1]);
    return R(q2, r, s[2]);
  }

  static Vector3d LSL(const Vector3d& q0, const Vector3d& s, const Scalar r) {
    Vector3d q1 = L(q0, r, s[0]);
    Vector3d q2 = S(q1, s[1]);
    return L(q2, r, s[2]);
  }

  static Vector3d RSR(const Vector3d& q0, const Vector3d& s, const Scalar r) {
    Vector3d q1 = R(q0, r, s[0]);
    Vector3d q2 = S(q1, s[1]);
    return R(q2, r, s[2]);
  }

  static Vector3d LSR(const Vector3d& q0, const Vector3d& s, const Scalar r) {
    Vector3d q1 = L(q0, r, s[0]);
    Vector3d q2 = S(q1, s[1]);
    return R(q2, r, s[2]);
  }

  static Vector3d RSL(const Vector3d& q0, const Vector3d& s, const Scalar r) {
    Vector3d q1 = R(q0, r, s[0]);
    Vector3d q2 = S(q1, s[1]);
    return L(q2, r, s[2]);
  }

  /// Given an initial dubins state (position, heading), a path primitive
  /// composed of three arc segments, and a turning radious, computes and
  /// returns the final dubins state.
  static Vector3d solve(const Vector3d q0, const Path_t& path, const Scalar r) {
    switch (path.id) {
      case dubins::LRLa:
      case dubins::LRLb:
        return LRL(q0, path.s, r);

      case dubins::RLRa:
      case dubins::RLRb:
        return RLR(q0, path.s, r);

      case dubins::LSL:
        return LSL(q0, path.s, r);

      case dubins::RSR:
        return RSR(q0, path.s, r);

      case dubins::LSR:
        return LSR(q0, path.s, r);

      case dubins::RSL:
        return RSL(q0, path.s, r);

      default:
        return Vector3d::Zero();
    }
  }
};

/// @see Integrate::solve
/**
 *  @tparam Scalar  number Scalar (double, float, etc)
 */
template <typename Scalar>
struct IntegrateIncrementally {
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3d;
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2d;
  typedef Path<Scalar> Path_t;

  static Vector3d L(const Vector3d& q0, const Scalar r, Scalar arc,
                    Scalar* budget) {
    if (*budget < r * arc) {
      arc = *budget / r;
    }
    *budget -= r*arc;
    Vector2d c = leftCenter(q0, r);
    Scalar theta1 = clampRadian(q0[2] + arc);
    Scalar alpha1 = leftAngleOf(theta1);

    return Vector3d(c[0] + r * std::cos(alpha1), c[1] + r * std::sin(alpha1),
                    theta1);
  }

  static Vector3d R(const Vector3d& q0, const Scalar r, Scalar arc,
                    Scalar* budget) {
    if (*budget < r * arc) {
      arc = *budget / r;
    }
    *budget -= r*arc;
    Vector2d c = rightCenter(q0, r);
    Scalar theta1 = clampRadian(q0[2] - arc);
    Scalar alpha1 = rightAngleOf(theta1);

    return Vector3d(c[0] + r * std::cos(alpha1), c[1] + r * std::sin(alpha1),
                    theta1);
  }

  static Vector3d S(const Vector3d& q0, Scalar d1, double* budget) {
    if(*budget < d1) {
      d1 = *budget;
    }
    *budget -= d1;
    return Vector3d(q0[0] + d1 * std::cos(q0[2]),
                    q0[1] + d1 * std::sin(q0[2]),
                    q0[2]);
  }

  static Vector3d LRL(const Vector3d& q0, const Vector3d& s, const Scalar r,
                      double* budget) {
    Vector3d q1 = L(q0, r, s[0], budget);
    Vector3d q2 = R(q1, r, s[1], budget);
    return L(q2, r, s[2], budget);
  }

  static Vector3d RLR(const Vector3d& q0, const Vector3d& s, const Scalar r,
                      double* budget) {
    Vector3d q1 = R(q0, r, s[0], budget);
    Vector3d q2 = L(q1, r, s[1], budget);
    return R(q2, r, s[2], budget);
  }

  static Vector3d LSL(const Vector3d& q0, const Vector3d& s, const Scalar r,
                      double* budget) {
    Vector3d q1 = L(q0, r, s[0], budget);
    Vector3d q2 = S(q1, s[1], budget);
    return L(q2, r, s[2], budget);
  }

  static Vector3d RSR(const Vector3d& q0, const Vector3d& s, const Scalar r,
                      double* budget) {
    Vector3d q1 = R(q0, r, s[0], budget);
    Vector3d q2 = S(q1, s[1], budget);
    return R(q2, r, s[2], budget);
  }

  static Vector3d LSR(const Vector3d& q0, const Vector3d& s, const Scalar r,
                      double* budget) {
    Vector3d q1 = L(q0, r, s[0], budget);
    Vector3d q2 = S(q1, s[1], budget);
    return R(q2, r, s[2], budget);
  }

  static Vector3d RSL(const Vector3d& q0, const Vector3d& s, const Scalar r,
                      double* budget) {
    Vector3d q1 = R(q0, r, s[0], budget);
    Vector3d q2 = S(q1, s[1], budget);
    return L(q2, r, s[2], budget);
  }

  /// Given an initial dubins state (position, heading), a path primitive
  /// composed of three arc segments, and a turning radious, computes and
  /// returns the final dubins state.
  static Vector3d solve(const Vector3d q0, const Path_t& path, const Scalar r,
                        double budget) {
    switch (path.id) {
      case dubins::LRLa:
      case dubins::LRLb:
        return LRL(q0, path.s, r, &budget);

      case dubins::RLRa:
      case dubins::RLRb:
        return RLR(q0, path.s, r, &budget);

      case dubins::LSL:
        return LSL(q0, path.s, r, &budget);

      case dubins::RSR:
        return RSR(q0, path.s, r, &budget);

      case dubins::LSR:
        return LSR(q0, path.s, r, &budget);

      case dubins::RSL:
        return RSL(q0, path.s, r, &budget);

      default:
        return Vector3d::Zero();
    }
  }
};

} // namespace curves_eigen
} // namespace dubins
} // namespace mpblocks

#endif // MPBLOCKS_DUBINS_CURVES_EIGEN_INTEGRATOR_H_
