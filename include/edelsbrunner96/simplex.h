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

#ifndef EDELSBRUNNER96_SIMPLEX_H_
#define EDELSBRUNNER96_SIMPLEX_H_

#include <array>
#include <bitset>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <set>

#include <Eigen/Dense>

namespace edelsbrunner96 {
namespace simplex {

/// enum defining bits for various markings of a simplex
/**
 *  As we perform the different steps of the edelsbrunner algorithm, we will
 *  mark simplices for various different conditions. These markings are stored
 *  as a bitset, and these enums define which bit corresponds to a particular
 *  mark
 */
enum MarkedBits {
  ISC_OPENED, ///< queued in induced subcomplex search
  ISC_MEMBER, ///< is a member of the induced subcomplex
  BFS_QUEUED, ///< has been queud in breadth-first search
  RETIRED,    ///< simplex has been marked for removal
  FEATURE_WALK, ///< simplex has been touched during feature walk
  LINE_WALK, ///< simplex has been touched during line walk
  VISIBLE_HULL_WALK, ///< simplex has been touched during hull walk
  VISIBLE_HULL, ///< simpex is a member of the visible hull
  FUZZY_WALK, ///< simplex has been touched during a fuzzy walk
  REMOVED_FOR_INSERT, ///< simplex has been removed during insertion of a point
  NUM_BITS
};

}  // namespace simplex

/// A simplex in the triangulation. A simplex is the convex hull of Ndim+1
/// points in R^NDim.
/**
 *  @tparam Traits  traits class for simplices in the triangulation, must
 *                  define, at least
 *    * Scalar        number format, i.e. double, float
 *    * Point         point type, i.e. Eigen::Vector
 *    * PointRef      how to refer to a point in the point store, for instance
 *                    Point*
 *    * PointDeref    functor for dereferencing a PointRef into a Point&
 *    * Simplex       the derived class of this type, used for defining the
 *                    neighborhood in a triangulation
 *    * SimplexRef    how to refer to a simplex in the simplex store, for
 *                    instance Simplex*
 *    * SimplexDeref  functor for dereferencing a SimplexRef into a Simplex&
 *    * NDim          number of dimensions
 *
 *  As a matter of convenience (as well as an optimization) we
 *  index vertices, faces, and neighbor in a corresponding manner.
 *  The i'th face is opposite the i'th vertex. Likewise the i'th
 *  neighbor is the neighbor opposite the i'th face.
 *
 *  Furthermore, vertices are stored in sorted order.
 */
template<class Traits>
struct SimplexBase {
  typedef typename Traits::Scalar       Scalar;
  typedef typename Traits::Point        Point;
  typedef typename Traits::PointRef     PointRef;
  typedef typename Traits::Simplex      Simplex;
  typedef typename Traits::SimplexRef   SimplexRef;
  typedef typename Traits::Storage      Storage;

  /// type for marked bit flags
  typedef std::bitset<simplex::NUM_BITS> MarkedVec;

  MarkedVec marked;       ///< flags for marking
  // TODO(bialkowski): I think version number is overkill and we can add a
  // "retired" flag to the bitset instead.
  uint16_t    version;      ///< incremented when flipped

  ///< vertices sorted
  std::array<PointRef, Traits::NDim + 1> V;

  /// neighbors, mapped to vertices
  std::array<SimplexRef, Traits::NDim + 1> N;

  Point       c;            ///< circumcenter
  Scalar      r2;           ///< circumradius squared

  /// initializes version and flags; in debug mode, also zeros out arrays
  SimplexBase();

  /// Set the vertices of a simplex directly from a list of point refs
  template <typename... PointRefs>
  void SetVertices(PointRefs... refs);

  /// Set the vertices of a simplex directly from a container of pointrefs
  template <typename Container>
  void SetVertices(Container refs);

  /// Set the vertices of a simplex directly from an initializer list of
  /// pointrefs
  void SetVertices(std::initializer_list<PointRef> refs);

  /// return the neighbor across the specified vertex
  /**
   *  binary searches V for the correct index
   */
  SimplexRef NeighborAcross(PointRef q) const;

  /// set the neighbor across the specified vertex
  /**
   *  binary searches V for the correct index
   */
  void SetNeighborAcross(PointRef q, SimplexRef n);

  /// return the index of the specified vertex
  uint8_t IndexOf(PointRef v) const;

  /// Given a simplex as the convex hull of the points in V, compute the
  /// coordinates of the point xq on the basis defined by
  /// [ (v[1] - v[0]), (v[2] - v[0]), ... (v[n] - v[0]) ]
  Eigen::Matrix<typename Traits::Scalar, Traits::NDim, 1>
    CoordinatesOf(Storage& storage, const typename Traits::Point& xq) const;

  /// Given a simplex as the convex hull of the points in V, compute the
  /// barycentric coordinates of the point xq. That is, find the vector
  /// lambda such that [v[0], v[1], ... v[n]][lambda] = xq and such that
  /// the elements of lambda sum to unity.
  Eigen::Matrix<typename Traits::Scalar, Traits::NDim + 1, 1>
  BarycentricCoordinates(Storage& storage,
                         const typename Traits::Point& xq) const;

  /// Interference test, return true if the query point lies inside the
  /// closed hull of the point set.
  bool Contains(Storage& storage, const typename Traits::Point& xq) const;

  /// called after vertices are filled computes the circumcenter
  void ComputeCenter(Storage& storage);
};

/// A convenience for setting vertices of a simplex
template <class Traits>
class SimplexFill {
 public:
  typedef typename Traits::Storage  Storage;
  typedef typename Traits::PointRef PointRef;
  typedef typename Traits::Simplex  Simplex;

  SimplexFill(Simplex& s)
      : index_(0),
        simplex_(s) {
  }

  void PushBack(PointRef p) {
    simplex_.V[index_++] = p;
  }

 private:
  uint8_t  index_;
  Simplex& simplex_;
};

/// Given a query point, @p x and a set of vertex points, V, return
/// the point y in hull(V) which is closest to @p x_q
/**
 *  @param  storage   the storage model
 *  @param  x         the query point
 *  @param  begin     iterator to the first vertex reference
 *  @param  end       iterator to the past-the-end vertex reference
 */
template <class Traits, class Derived, class Iterator>
void BarycentricProjection(
    typename Traits::Storage& storage, const typename Traits::Point& x,
    Iterator begin, Iterator end,
    Eigen::MatrixBase<Derived>* L,
    typename Traits::Point* x_proj);

/// Given a set of <= NDim+1 vertices, compute the distance of the query point
/// to the convex hull of the vertex set
/**
 *  We use a reduced version of GJK to compute the distance to the hull. We
 *  start at the nearest vertex and build up *features* until we find the
 *  nearest feature. A feature is a vertex, a line segment between two vertices,
 *  a triangle between three vertices, a tetrahedron between four vertices,
 *  etc.
 *
 *  @param  storage   the triangulation storage class
 *  @param  point     the query point
 *  @param  V         the vertex set of the simplex
 *  @param  out       output iterator, the vertex set of the nearest feature
 *                    is written here
 */
template <class Traits, class OutputIterator>
typename Traits::Scalar SimplexDistance(
    typename Traits::Storage& storage,
    const typename Traits::Point& x,
    const typename Traits::SimplexRef s_ref,
    typename Traits::Point* x_proj,
    OutputIterator V_out);

/// Computes the distance of a point to a simplex by expanding all combinations
/// of vertices and evaluating the distance to the lower-dimensional simplex
/// feature represented by that combination of vertices.
template <class Traits>
class ExhaustiveSimplexDistance {
 public:
  typedef typename Traits::Storage  Storage;
  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::Point  Point;
  typedef typename Traits::PointRef  PointRef;

  ExhaustiveSimplexDistance() {
    V_feature_.reserve(Traits::NDim+1);
    V_best_feature_.reserve(Traits::NDim+1);
    min_distance_ = std::numeric_limits<Scalar>::max();
  }

  template <typename InputIterator>
  ExhaustiveSimplexDistance<Traits>& Compute(Storage& storage, const Point& x,
                                             InputIterator begin,
                                             InputIterator end) {
    for (InputIterator iter = begin; iter != end; iter++) {
      V_feature_.push_back(*iter);
      BarycentricProjection<Traits>(storage, x, V_feature_.begin(),
                                    V_feature_.end(), &L_, &x_proj_);
      if (L_.minCoeff() >= 0) {
        distance_ = (x_proj_ - x).squaredNorm();
        if (distance_ < min_distance_) {
          V_best_feature_ = V_feature_;
          x_best_proj_ = x_proj_;
          min_distance_ = distance_;
        }
      }

      Compute(storage, x, iter + 1, end);
      V_feature_.pop_back();
    }
    return *this;
  }

  template <typename OutputIterator>
  void GetResult(Scalar* distance, Point* x_proj, OutputIterator V_out) {
    *distance = std::sqrt(min_distance_);
    *x_proj = x_best_proj_;
    std::copy(V_best_feature_.begin(), V_best_feature_.end(), V_out);
  }

 private:
  std::vector<PointRef> V_feature_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, Traits::NDim + 1, 1> L_;
  Point  x_proj_;
  Scalar distance_;

  std::vector<PointRef> V_best_feature_;
  Point x_best_proj_;
  Scalar min_distance_;
};

}  // namespace edelsbrunner96

#endif  // EDELSBRUNNER96_SIMPLEX_H_
