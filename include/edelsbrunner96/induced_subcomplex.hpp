/*
 *  Copyright (C) 2014 Josh Bialkowski (jbialk@mit.edu)
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
 *
 *  @date   Sept 17, 2014
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */
#ifndef EDELSBRUNNER96_INDUCED_SUBCOMPLEX_HPP_
#define EDELSBRUNNER96_INDUCED_SUBCOMPLEX_HPP_

#include <edelsbrunner96/induced_subcomplex.h>

#include <cassert>
#include <list>
#include <map>
#include <set>

#include <mpblocks/util/set_operations.hpp>

namespace edelsbrunner96 {

template<typename Traits>
void InducedSubcomplex<Traits>::Init(Storage& storage, SimplexRef s0_ref,
                                      SimplexRef s1_ref) {
  /* s0,s1 scope */{
    // convenient references for the following two set operations (scoped to
    // avoid confusion)
    Simplex& s0 = storage[s0_ref];
    Simplex& s1 = storage[s1_ref];

    // TODO(bialkowski): potential optimization: combine these two scans
    std::set_union(s0.V.begin(), s0.V.end(), s1.V.begin(), s1.V.end(),
                   V.begin());

    // Note: e[0] is the vertex missing from s0 and e[1] is the vertex
    // missing from s1, which is why e+1 comes before e+0 in the output
    // iterators
    set::IntersectionAndDifference(s0.V.begin(), s0.V.end(), s1.V.begin(),
                                   s1.V.end(), e.begin() + 1, e.begin(),
                                   f.begin());
  }

  // queue up the first two simples
  s[0] = s0_ref;
  s[1] = s1_ref;
}

template <typename Traits>
void InducedSubcomplex<Traits>::Init(Storage& storage,
                                     const std::array<SimplexRef, 2>& s) {
  Init(storage, s[0], s[1]);
}

template<class Traits>
bool InducedSubcomplex<Traits>::IsLocallyRegular(Storage& storage) {
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> VectorN;

  // if the other point is not in this sphere then it is locally reguarly,
  // otherwise it is not
  bool isRegular[2];
  for (int i = 0; i < 2; i++) {
    VectorN v = storage[e[i]] - storage[s[i]].c;
    isRegular[i] = v.squaredNorm() > storage[s[i]].r2;
  }

  return isRegular[0] && isRegular[1];
}

template<typename Traits>
void InducedSubcomplex<Traits>::Build(Storage& storage) {
  // initialize S with null references
  for (int i = 0; i < Traits::NDim + 2; i++) {
    S[i] = storage.NullSimplex();
  }

  // the set of simplices queued for expansion
  std::list<BreadthFirstNode> bfs_queue;

  // the set of simplices that have been expanded (so that we can clear the
  // mark)
  std::list<SimplexRef> evaluated_list;

  for (int i = 0; i < 2; i++) {
    SimplexRef s_ref = s[i];
    Simplex& s = storage[s_ref];
    PointRef v_missing = e[i];
    uint8_t i_of_missing =
        std::lower_bound(V.begin(), V.end(), v_missing) - V.begin();
    assert(i_of_missing < Traits::NDim + 2);
    s.marked[simplex::ISC_OPENED] = true;
    evaluated_list.push_back(s_ref);
    bfs_queue.emplace_back(s_ref, v_missing);
    S[i_of_missing] = s_ref;
  }

  while (bfs_queue.size() > 0) {
    BreadthFirstNode bfs_node = bfs_queue.front();
    bfs_queue.pop_front();

    for (int i = 0; i < Traits::NDim + 1; i++) {
      // Since neighbor is a simplex adjacent to bfs_node.s_ref, it shares
      // all but one vertex with bfs_node.s_ref. If that vertex it doesn't
      // share is the one of the facet that bfs_node.s_ref is missing, then
      // neighbor is also in the induced subcomplex, and the vertex it is
      // missing is s_ref.V[i].
      SimplexRef neighbor = storage[bfs_node.s_ref].N[i];
      PointRef v_missing = storage[bfs_node.s_ref].V[i];

      // TODO(bialkowski): If we choose to deal with boundary simplices by
      // null simplex references, then we need to watch out for them here
//      if( neighbor == storage.NullSimplex() ) {
//        continue;
//      }

      Simplex& s_neighbor = storage[neighbor];
      if (!s_neighbor.marked[simplex::ISC_OPENED]) {
        s_neighbor.marked[simplex::ISC_OPENED] = true;
        evaluated_list.push_back(neighbor);

        // as mentioned above, if we can find the vertex that bfs_node.s_ref is
        // missing in s_neighbor then s_neighbor is also in the induced complex
        if (std::binary_search(s_neighbor.V.begin(), s_neighbor.V.end(),
                               bfs_node.v_missing)) {
          // the index in induced_subcomplex.V of the vertex that is missing
          // from s_neighbor
          uint8_t i_of_missing =
              std::lower_bound(V.begin(), V.end(), v_missing) - V.begin();
          assert(i_of_missing < Traits::NDim + 2);
          S[i_of_missing] = neighbor;
          bfs_queue.emplace_back(neighbor, v_missing);
        }
      }
    }
  }

  // clear makers
  for (SimplexRef s_ref : evaluated_list) {
    storage[s_ref].marked[simplex::ISC_OPENED] = false;
  }
}

template <typename T> int signum(T val) {
    return (T(0) < val) - (val < T(0));
}

template<typename Traits>
bool InducedSubcomplex<Traits>::IsFlippable(Storage& storage) {
  typedef typename Traits::Scalar Scalar;
  typedef Eigen::Matrix<Scalar, Traits::NDim, Traits::NDim> Matrix;
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> Vector;

  // pre-fill the matrix used for computing hyperplanes
  Matrix A;
  for (int i = 0; i < Traits::NDim; i++) {
    A.row(i) = storage[f[i]];
  }

  // and the offset vector, the only requirement is that all the elements
  // of b are the same, (i.e. all points are the same distance from the
  // hyperplane along the normal), however by using the max coefficient of A
  // I have the vague notion that solving the system will have better
  // numerical stability
  Vector b;
  b.fill(1.0);

  // for each (NDim-1)simplex on the edge of the facet
  for (int i = 0; i < Traits::NDim; i++) {
    // Create the hyperplane composed of this edge and one of the
    // non-facet vertices (i.e. endpoints) of the simplex pair

    // Start by replacing the i'th row of A with one of the endpoints. We
    // arbitrarily choose e[0], though e[1] would work as well, as this test
    // is symmetric.
    A.row(i) = storage[e[0]];

    // We will compute the normal and offset from the removed facet point
    Matrix B;
    for(int j=0; j < Traits::NDim; j++) {
      B.row(j) = storage[f[i]];
    }

    // Now solve for the normal and offset of the hyperplane coincident to
    // this facet.
    Vector n = (A-B).fullPivLu().solve(b).normalized();
    Scalar o = (storage[e[0]] - storage[f[i]]).dot(n);
    if( o < 0 ) {
      n = -n;
      o = -o;
    }

    // now restore the row of A that we replaced
    A.row(i) = storage[f[i]];

    // Now check if the two other vertices of V are on the same side of this
    // facet, (i.e. the facet is a hull facet of V). If they're both on the
    // same side then the edge is convex and we may continue
    if (n.dot(storage[e[1]] - storage[f[i]]) < o) {
      continue;
    }

    // if they're not on the same side, then the edge is reflex, and
    // we must verify that it is incident only to members of the induced
    // subcomplex, this is the same as saying the S0 and S1 share the
    // same neighbor across from the point V[i] and that neighbor is
    // in the induced subcomplex.
    SimplexRef neighbors[2];
    for (int j = 0; j < 2; j++) {
      neighbors[j] = storage[s[j]].NeighborAcross(f[i]);
    }

    // If the two simplices do not share the neighbor across this edge then
    // the facet is not flippable;
    if (neighbors[0] != neighbors[1]) {
      return false;
    }

    // If the two share a neighbor across this point, then that neighbor
    // must be part of the induced subcomplex. This is because the neighbor
    // simplex is formed by d-1 edge vertices of the facet plus the two
    // endpoints (one from each bounding simplex). Therefore it must be
    // part of the induced subcomplex, and we do not even need to check if
    // neighbor is in S;
  }

  return true;
}

template<typename Traits>
uint8_t InducedSubcomplex<Traits>::FindIndexOfPeakVertex(
    typename Traits::Simplex& s) {
  uint8_t i = 0;
  uint8_t j = 0;

  while (i < Traits::NDim + 1 && j < Traits::NDim + 2) {
    while (V[j] < s.V[i]) {
      j++;
    }

    if (s.V[i] < V[j]) {
      return i;
    }
    ++i;
    ++j;
  }

  return i;
}

/*
 *  There are two ways we can accomplish the flip. The geometric way, and
 *  the graph way.
 *
 *  The geometric way:
 *  1) lift and invert all the points
 *  2) for each lifted d+1 subset of points determine the hyperplane in
 *     R^(d+1) that is coincident to those points
 *  3) use the "other" point to orient the face away from the simplex
 *  4) determine if the center of inversion is visible or not from that
 *     face
 *  5) separate the set of faces into visible/not-visible
 *  6) determine which one corresponds to our current triangulation
 *  7) replace it with the new one
 *
 *  The graph way (not particularly intuitive, but very clean):
 *  There are only NDim+2 possible simplices that can be formed by a set of
 *  NDim+2 vertices (NDim+2 choose NDim+1). We already have some of them in
 *  the current triangulation (non-null elements of S), so we just go through
 *  and replace them with the  Ndim+2 - |S| ones that aren't there. So, we go
 *  through each combination of Ndim+1 vertices. If it is already a simplex in
 *  the current triangulation, then remove it. If it is not already a simplex,
 *  then add it to the triangulation.
 */
template<typename Traits>
void InducedSubcomplex<Traits>::Flip(Storage& storage) {
  // Before modifying any simplices, build the neighborhood of the induced
  // subcomplex. Each hull facet of the subcomplex is found by removing two
  // vertices from the hull. Because the subcomplex is flippable, any
  // neighbor of a simplex in the induced subcomplex is either unique to that
  // simplex or else a member of the subcomplex.

  // TODO(bialkowski): sparse map or dense map? pay with storage or time?
  Eigen::Array<SimplexRef, Traits::NDim + 2, Traits::NDim + 2> neighborhood;
  neighborhood.fill(storage.NullSimplex());

  for (int i = 0; i < Traits::NDim + 2; i++) {
    if (S[i] == storage.NullSimplex()) {
      continue;
    }
    // If the neighbor of S[i] across V[j] is part of the subcomplex, then it
    // must be missing vertex j, so it must be in slot S[j].
    for (int j = 0; j < Traits::NDim + 2; j++) {
      if(i == j) {
        continue;
      }

      // if S[j] is empty then the neighbor of S[i] across V[j] is not in the
      // induced subcomplex and is therefore a neighbor of the induced
      // subcomplex.
      if (S[j] == storage.NullSimplex()) {
        SimplexRef neighbor = storage[S[i]].NeighborAcross(V[j]);
        neighborhood(i, j) = neighbor;
        neighborhood(j, i) = neighbor;
      }
    }
  }

  // We simply iterate over S and flip each simplex
  for (int i = 0; i < Traits::NDim + 2; i++) {
    // If the simplex does not exist, then create it
    if (S[i] == storage.NullSimplex()) {
      // retrieve a new simplex object
      S[i] = storage.Promote();

      // it's vertices are composed of all vertices except for the i'th
      // Note(bialkowski): No need to sort since V itself is sorted.
      std::remove_copy_if(V.begin(), V.end(), storage[S[i]].V.begin(),
                          [this,i](PointRef x) {return x == V[i];});

      // now that it has all it's vertices we can compute the
      // circumcenter
      storage[S[i]].ComputeCenter(storage);

      // If the simplex exists, then destroy it
    } else {
      storage.Retire(S[i]);
      S[i] = storage.NullSimplex();
    }
  }

  // Now we assign neighbors. The simplex at S[i] is missing the vertex V[i].
  // It's j'th neighbor contains V[i], and is missing S[i].V[j]
  for (int i = 0; i < Traits::NDim + 2; i++) {
    if (S[i] == storage.NullSimplex()) {
      continue;
    }
    for (int j = 0; j < Traits::NDim + 2; j++) {
      if (i == j) {
        continue;
      }
      // if S[j] is not in the current triangulation then the neighbor across
      // from V[j] must be a neighbor of the induced subcomplex
      if (S[j] == storage.NullSimplex()) {
        SimplexRef neighbor = neighborhood(i, j);
        Simplex& s_neighbor = storage[neighbor];
        storage[S[i]].SetNeighborAcross(V[j], neighborhood(i, j));

        uint8_t k = FindIndexOfPeakVertex(s_neighbor);
        storage[neighbor].N[k] = S[i];
      } else {
        storage[S[i]].SetNeighborAcross(V[j], S[j]);
      }
    }
  }
}

}  // namespace edelsbrunner96

#endif  // EDELSBRUNNER96_INDUCED_SUBCOMPLEX_HPP_
