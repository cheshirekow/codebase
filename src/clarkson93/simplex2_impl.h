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
#ifndef CLARKSON93_SIMPLEX2_IMPL_H_
#define CLARKSON93_SIMPLEX2_IMPL_H_

#include <mpblocks/clarkson93.hpp>
#include <map>

namespace mpblocks {
namespace clarkson93 {

template <class Traits>
inline typename Traits::PointRef SimplexOps<Traits>::vertex(Simplex& S,
                                                            uint32_t i) {
  return S.V[i];
}

template <class Traits>
inline typename Traits::SimplexRef SimplexOps<Traits>::neighbor(Simplex& S,
                                                                uint32_t i) {
  return S.N[i];
}

template <class Traits>
inline typename Traits::SimplexRef SimplexOps<Traits>::neighborAcross(
    Simplex& S, PointRef v) {
  return S.N[indexOf(S, v)];
}

template <class Traits>
inline typename Traits::SimplexRef SimplexOps<Traits>::peakNeighbor(
    Simplex& S) {
  return S.N[S.iPeak];
}

template <class Traits>
inline typename Traits::PointRef SimplexOps<Traits>::peak(Simplex& S) {
  return S.V[S.iPeak];
}

template <class Traits>
inline bool SimplexOps<Traits>::isMember(Simplex& S, SimplexBits bit) {
  return S.sets[bit];
}

template <class Traits>
inline uint32_t SimplexOps<Traits>::indexOf(Simplex& S, PointRef v) {
  return std::lower_bound(S.V, S.V + NDim + 1, v) - S.V;
}

template <class Traits>
inline NeighborRange<Traits> SimplexOps<Traits>::neighborhood(Simplex& S) {
  return NeighborRange<Traits>(S);
}

template <class Traits>
inline VertexRange<Traits> SimplexOps<Traits>::vertices(Simplex& S) {
  return VertexRange<Traits>(S);
}

template <class Traits>
template <typename Output1, typename Output2, typename Output3>
inline void SimplexOps<Traits>::vsetSplit(Simplex& Sa, Simplex& Sb,
                                          Output1 a_only, Output2 b_only,
                                          Output3 common) {
  PointRef* first1 = Sa.V;
  PointRef* last1 = first1 + NDim + 1;
  PointRef* first2 = Sb.V;
  PointRef* last2 = first2 + NDim + 1;

  while (first1 != last1 && first2 != last2) {
    if (*first1 < *first2)
      *a_only = *first1++;
    else if (*first2 < *first1)
      *b_only = *first2++;
    else {
      *common++ = *first1++;
      ++first2;
    }
  }
  std::copy(first1, last1, a_only);
  std::copy(first2, last2, b_only);
}

template <class Traits>
template <typename Output>
inline void SimplexOps<Traits>::vsetIntersection(Simplex& Sa, Simplex& Sb,
                                                 Output intersect) {
  std::set_intersection(Sa.V, Sa.V + NDim + 1, Sb.V, Sb.V + NDim + 1,
                        intersect);
}

template <class Traits>
template <typename Input, typename Output>
inline void SimplexOps<Traits>::neighborSharing(Simplex& S, Input first,
                                                Input last, Output out) {
  for (unsigned int i = 0; i < NDim + 1; i++) {
    if (S.V[i] < *first)
      *out++ = S.N[i];
    else
      first++;
  }
}

template <class Traits>
inline Surrogate<typename Traits::PointRef> SimplexOps<Traits>::setPeak(
    Simplex& S) {
  return Surrogate<PointRef>(S.V[S.iPeak]);
}

template <class Traits>
inline Surrogate<typename Traits::PointRef> SimplexOps<Traits>::setVertex(
    Simplex& S, uint32_t i) {
  return Surrogate<PointRef>(S.V[i]);
}

template <class Traits>
inline Surrogate<typename Traits::SimplexRef> SimplexOps<Traits>::setNeighbor(
    Simplex& S, uint32_t i) {
  return Surrogate<SimplexRef>(S.N[i]);
}

template <class Traits>
inline Surrogate<typename Traits::SimplexRef>
SimplexOps<Traits>::setNeighborAcross(Simplex& S, PointRef v) {
  uint32_t i = std::lower_bound(S.V, S.V + NDim + 1, v) - S.V;
  return Surrogate<SimplexRef>(S.N[i]);
}

template <class Traits>
inline std::bitset<simplex::NUM_BITS>::reference SimplexOps<Traits>::setMember(
    Simplex& S, SimplexBits bit) {
  return S.sets[bit];
}

template <class Traits>
inline void SimplexOps<Traits>::finish(Simplex& S) {
  PointRef peak = S.V[S.iPeak];
  std::map<PointRef, SimplexRef> kv;

  for (int i = 0; i < NDim + 1; i++) kv[S.V[i]] = S.N[i];

  int i = 0;
  for (auto pair : kv) {
    if (pair.first == peak) S.iPeak = i;
    S.V[i] = pair.first;
    S.N[i] = pair.second;
    i++;
  }
}

template <class Traits>
inline void SimplexOps<Traits>::computeBase(Simplex& S, Deref& deref) {
  typedef Eigen::Matrix<Scalar, NDim, NDim> Matrix;
  typedef Eigen::Matrix<Scalar, NDim, 1> Vector;

  Matrix A;
  Vector b;

  unsigned int j = 0;
  for (unsigned int i = 0; i < NDim + 1; i++)
    if (i != S.iPeak) A.row(j++) = deref.point(S.V[i]);
  b.setConstant(1);

  // solve for the normal
  S.n = A.fullPivLu().solve(b);
  S.n.normalize();

  // and then find the value of 'c' (hyperplane offset)
  j = S.iPeak == 0 ? 1 : 0;
  S.o = deref.point(S.V[j]).dot(S.n);
}

template <class Traits>
inline void SimplexOps<Traits>::orientBase(Simplex& S, const Point& x,
                                           Orientation orient) {
  // then orient the hyperplane so that the peak vertex is on the
  // "positive" side (i.e. the 'less than' side)

  switch (orient) {
    case INSIDE: {
      if (S.n.dot(x) > S.o) {
        S.n = -S.n;
        S.o = -S.o;
      }
      break;
    }

    case OUTSIDE: {
      if (S.n.dot(x) < S.o) {
        S.n = -S.n;
        S.o = -S.o;
      }
      break;
    }

    default:
      assert(false);
      break;
  }
}

template <class Traits>
inline typename Traits::Scalar SimplexOps<Traits>::normalProjection(
    const Simplex& S, const Point& x) {
  return S.o - S.n.dot(x);
}

template <class Traits>
inline bool SimplexOps<Traits>::isInfinite(const Simplex& S,
                                           PointRef antiOrigin) {
  return S.V[S.iPeak] == antiOrigin;
}

template <class Traits>
inline bool SimplexOps<Traits>::isVisible(const Simplex& S, const Point& x) {
  return (S.n.dot(x) < S.o);
}

}  // namespace clarkson93
}  // namespace mpblocks

#endif  // CLARKSON93_SIMPLEX2_IMPL_H_
