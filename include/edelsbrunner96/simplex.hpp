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

#ifndef EDELSBRUNNER96_SIMPLEX_HPP_
#define EDELSBRUNNER96_SIMPLEX_HPP_

#include <edelsbrunner96/simplex.h>

#include <algorithm>
#include <mpblocks/util/set_operations.hpp>

namespace edelsbrunner96 {

template<class Traits>
SimplexBase<Traits>::SimplexBase() {
  version = 0;
  marked.reset();
#ifndef NDEBUG
  r2 = 0;
  c.fill(0);
#endif
}

/// A trick to allow parameter pack expansion for filling
inline void pass(std::initializer_list<int>&&) {}

template<class Traits>
template<typename ... PointRefs>
void SimplexBase<Traits>::SetVertices(PointRefs ... refs) {
  SimplexFill<Traits> filler(*this);
  // notes on esoteric syntax: fuller.PushBack must be expanded inside an
  // initializer list otherwise to guarentee that PushBack occurs in the
  // desired order. We must call wrap the output of filler.PushBack as a
  // (void,int) because it returns void, and we need to return *something* to
  // build the throw-away initializer list.
  pass({ (filler.PushBack(refs),1)... });
  std::sort(V.begin(), V.end());
}

template<class Traits>
template<typename Container>
void SimplexBase<Traits>::SetVertices(Container refs) {
  int index_ = 0;
  for(PointRef ref : refs) {
    V[index_++] = ref;
  }
  std::sort(V.begin(), V.end());
}

template<class Traits>
void SimplexBase<Traits>::SetVertices(std::initializer_list<PointRef> refs) {
  int index_ = 0;
  for(PointRef ref : refs) {
    V[index_++] = ref;
  }
  std::sort(V.begin(), V.end());
}

template <class Traits>
inline typename Traits::SimplexRef SimplexBase<Traits>::NeighborAcross(
    PointRef q) const {
  // note: lower_bound is binary search,
  return N[IndexOf(q)];
}

template<class Traits>
inline void SimplexBase<Traits>::SetNeighborAcross(PointRef q, SimplexRef n) {
  // note: lower_bound is binary search,
  N[IndexOf(q)] = n;
}

template<class Traits>
inline uint8_t SimplexBase<Traits>::IndexOf(PointRef v) const {
  // note: lower_bound is binary search,
  const auto iter = std::lower_bound(V.begin(), V.end(), v);
  return iter - V.begin();
}

template <class Traits>
Eigen::Matrix<typename Traits::Scalar, Traits::NDim, 1>
SimplexBase<Traits>::CoordinatesOf(Storage& storage,
                                   const typename Traits::Point& xq) const {
  typedef Eigen::Matrix<Scalar, Traits::NDim, Traits::NDim> Matrix;

  Matrix A;
  const Point& v0 = storage[V[0]];
  for (int i = 0; i < Traits::NDim; i++) {
    A.col(i) = storage[V[i + 1]] - v0;
  }

  // perform the LU decomposition
  Eigen::FullPivLU<Matrix> LU(A.transpose());
  // TODO(bialkowski): validate that the simplex is well-conditioned
  // Scalar det = LU.determinant();

  return LU.solve(xq - v0);
}

template <class Traits>
Eigen::Matrix<typename Traits::Scalar, Traits::NDim + 1, 1>
SimplexBase<Traits>::BarycentricCoordinates(
    Storage& storage, const typename Traits::Point& xq) const {
  typedef Eigen::Matrix<Scalar, Traits::NDim + 1, Traits::NDim + 1> Matrix;

  Matrix A;
  for (int i = 0; i < Traits::NDim + 1; i++) {
    A.template block<Traits::NDim, 1>(0, i) = storage[V[i]];
    A(Traits::NDim, i) = 1;
  }

  // perform the LU decomposition
  Eigen::FullPivLU < Matrix > LU(A);
  // TODO(bialkowski): validate that the simplex is well-conditioned
  // Scalar det = LU.determinant();

  Eigen::Matrix<Scalar, Traits::NDim + 1, 1> xq_raised;
  xq_raised.template head<Traits::NDim>() = xq;
  xq_raised[Traits::NDim] = 1;

  return LU.solve(xq_raised);
}

template<class Traits>
bool SimplexBase<Traits>::Contains(Storage& storage, const Point& xq) const {
  // If the point lies inside the hull, then all of the barycentric coordinates
  // are positive
  return this->BarycentricCoordinates(storage, xq).minCoeff() > 0;
}

template<class Traits>
void SimplexBase<Traits>::ComputeCenter(Storage& storage) {
  typedef Eigen::Matrix<Scalar, Traits::NDim, Traits::NDim> Matrix;
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> Vector;

  // calculate circumcenter of the simplex
  // see 2012-10-26-Note-11-41_circumcenter.xoj for math
  Matrix A;
  Vector b;

  const Point& v0 = storage[V[0]];
  for (int i = 0; i < Traits::NDim; i++) {
    Vector dv = storage[V[i + 1]] - v0;
    A.row(i) = 2*dv;
    b[i] = storage[V[i + 1]].squaredNorm() - v0.squaredNorm();
  }

  // the circum center
  c = A.fullPivLu().solve(b);

  // squared radius of the circumsphere
  r2 = (c - v0).squaredNorm();
}

template <class Traits, class Derived, class Iterator>
void BarycentricProjection(
    typename Traits::Storage& storage, const typename Traits::Point& x,
    Iterator begin, Iterator end,
    Eigen::MatrixBase<Derived>* L,
    typename Traits::Point* x_proj) {

  typedef typename Traits::Scalar Scalar;
  typedef Eigen::Matrix<Scalar, Traits::NDim, Eigen::Dynamic> Matrix;
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> Vector;
  typedef typename Matrix::Index Index;

  // number of elements
  Index n = (end - begin);
  assert(n >= 1);
  assert(n <= Traits::NDim + 1);

  Vector v0 = storage[*begin++];
  if (n == 1) {
    L->derived().resize(1,1);
    (*L)[0] = Scalar(1.0);
    *x_proj = v0;
    return;
  }

  Matrix V(int(Traits::NDim), n - 1);
  L->derived().resize(n, 1);
  for (int i = 0; begin != end; ++begin, ++i) {
    V.col(i) = (storage[*begin] - v0);
  }
  auto full_piv_lu = (V.transpose() * V).fullPivLu();
  L->template tail(n-1) = full_piv_lu.solve(V.transpose() * (x - v0));
  (*L)[0] = Scalar(1.0) - L->template tail(n-1).sum();
  *x_proj = V * (L->template tail(n - 1)) + v0;
}

template <class Traits, class OutputIterator>
typename Traits::Scalar SimplexDistance(typename Traits::Storage& storage,
                                        const typename Traits::Point& x,
                                        const typename Traits::SimplexRef s_ref,
                                        typename Traits::Point* x_proj,
                                        OutputIterator V_out) {
  typedef typename Traits::PointRef PointRef;
  typedef typename Traits::Point Point;
  typedef typename Traits::Scalar Scalar;

  // local copy of the vertex set which we will sort as we go through the
  // GJK algorithm.
  std::array<PointRef, Traits::NDim + 1> V;
  std::copy(storage[s_ref].V.begin(), storage[s_ref].V.end(), V.begin());
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> L;

  // Build up the nearest feature, starting with a single vertex
  auto begin = V.begin();
  if(*begin == storage.NullPoint()) {
    ++begin;
  }

  auto end = begin+1;
  for (; end <= V.end(); ++end) {
    assert(begin < end);
    // compute the distance to the current feature
    BarycentricProjection<Traits>(storage, x, begin, end, &L, x_proj);

    // if the projection does not lie inside the sub-simplex, then start
    // over with the last vertex found
    if (L.minCoeff() < 0) {
      std::swap(*begin, *(end-1));
      end = begin+1;
      BarycentricProjection<Traits>(storage, x, begin, end, &L, x_proj);
    }
    Point normal = (*x_proj - x).normalized();

    if(end == V.end()) {
      std::sort(begin, end);
      std::copy(begin, end, V_out);
      return (*x_proj - x).norm();
    }

    // find the vertex that is not currently part of the feature which is
    // nearest in the direction x -> x_proj.
    auto iter_nearest = end;
    Scalar dist_nearest = (storage[*iter_nearest] - x).dot(normal);
    for (auto iter = iter_nearest + 1; iter != V.end(); ++iter) {
      // Note: this is a signed distance
      Scalar dist = (storage[*iter] - x).dot(normal);
      if (dist < dist_nearest) {
        iter_nearest = iter;
        dist_nearest = dist;
      }
    }

    // if the nearest point in the search direction is further than the
    // current feature point, then the nearest feature is built
    if (dist_nearest > (*x_proj - x).dot(normal)) {
      // copy out the feature (as a vertex set)
      std::sort(begin, end);
      std::copy(begin, end, V_out);
      return (*x_proj - x).norm();
    } else {
      // otherwise move the nearest vertex in the search direction into the
      // set of vertices describing the nearest feature
      std::swap(*end, *iter_nearest);
      continue;
    }
  }

  std::copy(begin, end, V_out);
  std::sort(begin, end);
  return (*x_proj - x).norm();
}

}  // namespace edelsbrunner96

#endif // EDELSBRUNNER96_SIMPLEX_HPP_
