/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of clarkson93.
 *
 *  clarkson93 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  clarkson93 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with clarkson93.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef CLARKSON93_SIMPLEX2_IMPL_H_
#define CLARKSON93_SIMPLEX2_IMPL_H_

#include <algorithm>
#include <map>
#include <clarkson93/simplex.h>

namespace clarkson93 {

template <class Traits, typename Output1, typename Output2, typename Output3>
inline void VsetSplit(const Simplex<Traits>& simplex_a,
                      const Simplex<Traits>& simplex_b, Output1 a_only,
                      Output2 b_only, Output3 intersect) {
  auto first1 = simplex_a.V.begin();
  auto last1 = simplex_a.V.end();
  auto first2 = simplex_b.V.begin();
  auto last2 = simplex_b.V.end();

  while (first1 != last1 && first2 != last2) {
    if (*first1 < *first2)
      *a_only = *first1++;
    else if (*first2 < *first1)
      *b_only = *first2++;
    else {
      *intersect++ = *first1++;
      ++first2;
    }
  }
  std::copy(first1, last1, a_only);
  std::copy(first2, last2, b_only);
}

template <class Traits, typename Output>
void VsetIntersection(const Simplex<Traits>& simplex_a,
                      const Simplex<Traits>& simplex_b, Output intersect) {
  std::set_intersection(simplex_a.V.begin(), simplex_a.V.end(),
                        simplex_b.V.begin(), simplex_b.V.end(), intersect);
}

template <class Traits, class Container, class Output>
void GetNeighborsSharing(const Simplex<Traits>& simplex,
                         const Container& feature, Output out_iter) {
  auto facet_iter = feature.begin();
  for (unsigned int i = 0; i < Traits::kDim + 1 && facet_iter != feature.end();
       i++) {
    if (simplex.V[i] < *facet_iter) {
      *out_iter++ = simplex.N[i];
    } else {
      ++facet_iter;
    }
  }
}

template <class Traits>
void SortVertices(Simplex<Traits>* simplex) {
  typename Traits::PointRef peak = simplex->GetPeakVertex();

  // TODO(josh): pqueue might be faster
  std::map<typename Traits::PointRef, Simplex<Traits>*> kv;

  for (int i = 0; i < Traits::kDim + 1; i++) {
    kv[simplex->V[i]] = simplex->N[i];
  }

  int i = 0;
  for (auto pair : kv) {
    if (pair.first == peak) {
      simplex->i_peak = i;
    }
    simplex->V[i] = pair.first;
    simplex->N[i] = pair.second;
    i++;
  }
}

template <class Traits, class Deref>
void ComputeBase(Simplex<Traits>* simplex, const Deref& deref) {
  typedef Eigen::Matrix<typename Traits::Scalar, Traits::kDim, Traits::kDim>
      Matrix;
  typedef Eigen::Matrix<typename Traits::Scalar, Traits::kDim, 1> Vector;

  Matrix A;
  Vector b;

  int j = 0;
  for (int i = 0; i < Traits::kDim + 1; i++)
    if (i != simplex->i_peak)
      A.row(j++) = deref(simplex->V[i]);
  b.setConstant(1);

  // solve for the normal
  simplex->n = A.fullPivLu().solve(b).normalized();

  // and then find the value of 'c' (hyperplane offset)
  j = simplex->i_peak == 0 ? 1 : 0;
  simplex->o = deref(simplex->V[j]).dot(simplex->n);
}

template <class Traits, class Point>
void OrientBase(Simplex<Traits>* simplex, const Point& x,
                simplex::Orientation orient) {
  // orient the hyperplane so that the peak vertex is on the
  // "positive" side (i.e. the 'less than' side)
  switch (orient) {
    case simplex::INSIDE: {
      if (simplex->n.dot(x) > simplex->o) {
        simplex->n = -simplex->n;
        simplex->o = -simplex->o;
      }
      break;
    }

    case simplex::OUTSIDE: {
      if (simplex->n.dot(x) < simplex->o) {
        simplex->n = -simplex->n;
        simplex->o = -simplex->o;
      }
      break;
    }

    default:
      assert(false);
      break;
  }
}

template <class Traits, class Point>
typename Traits::Scalar NormalProjection(const Simplex<Traits>& simplex,
                                         const Point& x) {
  return simplex->o - simplex->n.dot(x);
}

template <class Traits>
bool IsInfinite(const Simplex<Traits>& simplex,
                typename Traits::PointRef anti_origin) {
  return simplex->V[simplex->i_peak] == anti_origin;
}

template <class Traits, class Point>
bool IsVisible(const Simplex<Traits>& simplex, const Point& x) {
  return (simplex->n.dot(x) < simplex->o);
}

}  // namespace clarkson93

#endif  // CLARKSON93_SIMPLEX2_IMPL_H_
