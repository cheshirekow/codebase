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

#ifndef MPBLOCKS_EDELSBRUNNER96_TRIANGULATION_HPP_
#define MPBLOCKS_EDELSBRUNNER96_TRIANGULATION_HPP_

#include <edelsbrunner96/triangulation.h>

#include <algorithm>
#include <edelsbrunner96/simplex.hpp>
#include <edelsbrunner96/induced_subcomplex.hpp>
#include <edelsbrunner96/line_walker.hpp>
#include <mpblocks/util/set_operations.hpp>

namespace edelsbrunner96 {

template <class Traits>
typename Traits::SimplexRef Triangulate(
    typename Traits::Storage& storage,
    std::initializer_list<typename Traits::PointRef> refs) {
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::Simplex Simplex;

  // there must be NDim + 1 points in the point set to create the initial
  // simplices
  assert(refs.size() == Traits::NDim + 1);

  SimplexRef simplex_refs[Traits::NDim + 1];
  SimplexRef center_ref = storage.Promote();

  // we must construct simplices before assigning neighbors so we can
  // point to actual addresses during neighborhood mapping
  for (int i = 0; i < Traits::NDim + 1; i++) {
    simplex_refs[i] = storage.Promote();
  }

  // Note: it is now safe to grab the reference since we are not creating
  // any new simplices and the underlying storage may not move out from under
  // us.
  Simplex& center_simplex = storage[center_ref];
  std::copy(refs.begin(), refs.end(), center_simplex.V.begin());

  for (int i = 0; i < Traits::NDim + 1; i++) {
    auto pr = storage[simplex_refs[i]].V.begin();
    auto sr = storage[simplex_refs[i]].N.begin();

    *pr++ = storage.NullPoint();
    *sr++ = center_ref;

    for (int j = 0; j < Traits::NDim + 1; j++) {
      if (i == j) {
        continue;
      }
      *pr++ = center_simplex.V[j];
      *sr++ = simplex_refs[j];
    }

    center_simplex.N[i] = simplex_refs[i];
  }

  return center_ref;
}

template <class Traits>
typename Traits::SimplexRef Maintain(typename Traits::Storage& storage,
                                     typename Traits::PointRef point_ref,
                                     typename Traits::SimplexRef return_ref,
                                     std::list<Facet<Traits>>& link_facets) {
  typedef typename Traits::SimplexRef SimplexRef;
  typedef Facet<Traits> Facet;

  // while the link stack is not empty, flip any non-regular flippable facets
  while (!link_facets.empty()) {
    Facet facet = link_facets.back();
    link_facets.pop_back();

    // if the facet no longer exists (due to some previous flip) then simply
    // ignore it
    if (!facet.StillExists(storage)) {
      continue;
    }

    // if the facet exists but it is already locally regular then we do not
    // need to do anything
    InducedSubcomplex<Traits> induced_subcomplex;
    induced_subcomplex.Init(storage, facet.s);
    if (induced_subcomplex.IsLocallyRegular(storage)) {
      continue;
    }

    // If it is not locally regular we must build the induced subcomplex of
    // the facet and test if it is flippable (i.e. all reflex edges are filled
    // by another simplex in the induced subcomplex). If it is not flippable
    // then simply move on.
    induced_subcomplex.Build(storage);
    if (!induced_subcomplex.IsFlippable(storage)) {
      continue;
    }

    // If the facet is locally non-regular and flippable then flip it
    induced_subcomplex.Flip(storage);
    for (SimplexRef s_ref : induced_subcomplex.S) {
      if (s_ref != storage.NullSimplex()) {
        return_ref = s_ref;
      }
    }

    // After the flip, there can be at most one simplex which does not
    // contain point_ref in it's vertex set. That simplex is stored in
    // induced_subcomplex.S at the same index of  as point_ref in
    // induced_subcomplex.V.

    // Additionally, any neighbor of induced subcomplex is a neighbor of
    // exactly one simplex inside the induced subcomplex.

    // Therefore all new simplices which contain point_ref in it's vertex
    // set have one facet which is part of the link of the inserted point.
    // This facet is the one across from point_ref.

    typedef std::pair<SimplexRef, SimplexRef> SimplexPair;
    std::set<SimplexPair> facet_pairs;
    for (int i = 0; i < Traits::NDim + 2; i++) {
      // if i is the index of point_ref in the induced subcomplex then S[i],
      // even if it is not null, is the one simplex which does not contain
      // point_ref and is therefore not part of the link, except perhaps as
      // the neighbor of a simplex which is inside the link surface.
      if (induced_subcomplex.V[i] == point_ref) {
        continue;
      }

      // skip null simplices
      SimplexRef link_simplex = induced_subcomplex.S[i];
      if (link_simplex == storage.NullSimplex()) {
        continue;
      }

      // find the neighbor across from point_ref
      SimplexRef neighbor = storage[link_simplex].NeighborAcross(point_ref);
      // skip any link facets which are on the hull of the point set
      if (storage[neighbor].V[0] == storage.NullPoint()) {
        continue;
      }
      link_facets.emplace_back();
      link_facets.back().Construct(storage, link_simplex, neighbor);
    }
  }

  return return_ref;
}


/// For each infinite simplex references in @p hull_simplices, replace the
/// null vertex with the vertex pointed to by @p point_ref
template <class Traits, class Container>
void DemoteHullSimplices(typename Traits::Storage& storage,
                         Container& visible_hull,
                         typename Traits::PointRef point_ref) {
  typedef typename Traits::Simplex Simplex;
  typedef typename Traits::SimplexRef SimplexRef;

  // first, demote all hull simplices by inserting point_ref into their
  // vertex set while maintaining all neighbor relationships (i.e. resorting
  // the V and S sets).
  for (SimplexRef s_ref : visible_hull) {
    /* Scope-protect s,N,V */ {
      Simplex& s = storage[s_ref];
      assert(s.V.end() - s.V.begin() > 1);

      SimplexRef neighbor_across = s.N[0];

      auto N = s.N.begin();
      auto V = s.V.begin();

      // Shift all vertex and neighbor references to the left by one slot until
      // we find the slot where the new reference belongs. Note: if all vertex
      // references are less than the new vertex reference, then at the end
      // of this loop V will point to s.V + NDim which is the last slot in the
      // vertex set.
      for (; V != s.V.end()-1 && *(V + 1) < point_ref; ++V, ++N) {
        *V = *(V + 1);
        *N = *(N + 1);
      }

      *V = point_ref;
      *N = neighbor_across;
      s.ComputeCenter(storage);
    }
  }
}

/// Build the horizon ridge, which is the set of all *edges* which border
/// the x-visible hull.
/**
 * An edge is a set of (N+1-2) vertices. For simplices
 * along the horizon ridge, one of the missing vertices is the infinite
 * vertex, or, if the simplex is newly demoted, the vertex that was added.
 * We go through all of the demoted simplices, and make a list of those that
 * are on the horizon ridge (i.e. the boundary of the demoted subcomplex).
 */
template <class Traits, class Container>
std::list<std::pair<typename Traits::SimplexRef, typename Traits::SimplexRef>>
BuildHorizonRidge(typename Traits::Storage& storage, Container& visible_hull) {
  typedef typename Traits::Simplex Simplex;
  typedef typename Traits::SimplexRef SimplexRef;
  typedef std::pair<SimplexRef, SimplexRef> Edge;

  std::list<Edge> horizon_ridge;
  for (SimplexRef s_ref : visible_hull) {
    Simplex& s = storage[s_ref];
    for (int i = 0; i < Traits::NDim + 1; i++) {
      SimplexRef neighbor_ref = s.N[i];
      if (storage[neighbor_ref].V[0] == storage.NullPoint() &&
          !storage[neighbor_ref].marked[simplex::VISIBLE_HULL]) {
        horizon_ridge.emplace_back(s_ref, neighbor_ref);
      }
    }
  }
  return horizon_ridge;
}

/// For each simplex pair (s1,s2) in @horizon_ridge, fill the empty wedge that
/// was created when s1 was demoted to a non-infinite simplex
template <class Traits, class Container>
std::list<typename Traits::SimplexRef> FillHorizonWedges(
    typename Traits::Storage& storage, Container& horizon_ridge,
    typename Traits::PointRef point_ref) {
  typedef typename Traits::Simplex Simplex;
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::PointRef PointRef;

  std::list<SimplexRef> horizon_wedges;
  for (auto& pair : horizon_ridge) {
    // Prior to demoting the first simplex in the pair, the two simplices
    // across the horizon facet shared all but one vertex. After demoting the
    // first simplex, they share all but two vertices.
    SimplexRef wedge_ref = storage.Promote();
    horizon_wedges.push_back(wedge_ref);
    Simplex& wedge_simplex = storage[wedge_ref];

    // The wedge simplex contains those (NDim+1-2) common vertices, as well as
    // the peak vertex, and the infinite vertex.
    wedge_simplex.V[0] = storage.NullPoint();
    std::set_intersection(
        storage[pair.first].V.begin(), storage[pair.first].V.end(),
        storage[pair.second].V.begin(), storage[pair.second].V.end(),
        wedge_simplex.V.begin() + 1);
    wedge_simplex.V[Traits::NDim] = point_ref;

    // If we may operate under the assumption that poin_ref, being a new point,
    // has higher value than all vertices before it (i.e. pre-allocated block
    // storage) then this step is unnecessary and can be optimized out.
    // TODO (bialkowski): add scheme to optimize out this step if traits is
    // sufficiently annotated.
    std::sort(wedge_simplex.V.begin(), wedge_simplex.V.end());

    // We will initialize all neighbors to Null
    for (int i = 0; i < Traits::NDim + 1; i++) {
      wedge_simplex.N[i] = storage.NullSimplex();
    }

    Simplex& s_a = storage[pair.first];
    Simplex& s_b = storage[pair.second];
    PointRef v_a[2] = {storage.NullPoint(), storage.NullPoint()};
    PointRef v_b[2] = {storage.NullPoint(), storage.NullPoint()};

    set::SymmetricDifference(s_a.V.begin(), s_a.V.end(), s_b.V.begin(),
                             s_b.V.end(), v_a, v_b);
    assert(v_a[0] == point_ref || v_a[1] == point_ref);
    assert(v_b[0] == storage.NullPoint() || v_b[1] == storage.NullPoint());

    PointRef v_a_across_b = v_a[0] == point_ref ? v_a[1] : v_a[0];
    PointRef v_b_across_a = v_b[0] == storage.NullPoint() ? v_b[1] : v_b[0];
    assert(v_a_across_b != point_ref);
    assert(v_a_across_b != storage.NullPoint());
    assert(v_b_across_a != storage.NullPoint());
    assert(s_a.NeighborAcross(v_a_across_b) == pair.second);
    assert(s_b.NeighborAcross(v_b_across_a) == pair.first);

    // We know exactly two neighbors of the wedge simplex at this point. Also,
    // some of the simplex's other neighbors haven't ben alloc'ed yet
    wedge_simplex.SetNeighborAcross(point_ref, pair.second);
    wedge_simplex.SetNeighborAcross(storage.NullPoint(), pair.first);
    int i_first = s_a.IndexOf(v_a_across_b);
    assert(s_a.N[i_first] == pair.second);
    s_a.N[i_first] = wedge_ref;
    int i_second = s_b.IndexOf(v_b_across_a);
    assert(s_b.N[i_second] == pair.first);
    s_b.N[i_second] = wedge_ref;
  }

  return horizon_wedges;
}

/// Given a simplex which fills an empty wedge created by demotion of a
/// simplex along the horizon ridge, find the i'th neighbor by walking around
/// the common edge which contains the peak vertex
/**
 * The algorithm for this is based on the horizon-ridge algorithm from
 * Clarkson93. For each new simplex along the horizon ridge that we need to
 * find the neighbor of, the neighbor that we need will lie across a *facet* of
 * that simplex. That facet is composed of N vertices, one of which is the peak
 * vertex (the newly added point). If we remove the peak vertex we are left
 * with (N-1) vertices which we will call the "pivot edge". We will do a walk
 * around this pivot edge until we find the neighboring simplex.

 * From our starting simplex we move to the neighbor across the infinite
 * vertex. This neighbor shares the pivot edge, and one other vertex in
 * common with the starting simplex. Incidentally, it is also a recently
 * demoted simplex. We continue the walk by following that
 * other vertex. A proof is given in the Clarkson93 paper that the walk
 * *must* end at the desired neighbor of the starting simplex.
 *
 * This walk is illustrated in 2d in the following ascii art. We start at
 * simplex [0] and are looking for the neighbor across the facet {x,inf}.
 * The relevant edge we will walk "around" is {x}. Note that we define an
 * edge as the NDim points of the common facet, minus the infinite vertex.
 * In 2d this is a set of one vertex, or, simply, a point. In 2d, an edge is
 * a point... confusing, I know.
 *
 * In any case we start at simplex [0] and walk to the neighbor across from
 * (inf) which is simplex [1] in the diagram. Simplex [1] contains exactly one
 * vertex which is on the facet between [0] and [1] and is not (x). We identify
 * that vertex (a), and then move to the neighbor across that vertex, [2].
 * Again simplex [2] and simplex [1] share only a single vertex which is on the
 * their common facet but is not (x). We identify that vertex (b) and move to
 * the neighbor across (b), in this case simplex [3]. We continue this walk
 * until we reach simplex [4], which is the first infinite simplex in this
 * walk around the edge. Simplex [4] is the neighbor across from [0] that
 * we were searching for.
 *
 *  @code
 *                (inf)           [.] simplex
 *                 |              (.) vertex
 *                 |
 *        [4]   (x)|     [0]
 *               //\\
 *            /  /  \  \
 *         /    /    \    \
 *      /      /      \      \
 *   /   [3]  /   [2]  \  [1]   \
 *  (d)      (c)       (b)      (a)
@endcode
 *
 */
template <class Traits>
void FindWedgeNeighbor(typename Traits::Storage& storage,
                       typename Traits::SimplexRef wedge_ref, int i) {
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::PointRef PointRef;

  // Copy all but the 0'th and i'th vertices
  PointRef V_edge[Traits::NDim - 1];
  /* scope of V */ {
    PointRef* V = V_edge;
    for (int j = 1; j < Traits::NDim + 1; j++) {
      if (j == i) {
        continue;
      } else {
        *V++ = storage[wedge_ref].V[j];
      }
    }
  }

  SimplexRef prev_ref = wedge_ref;
  SimplexRef next_ref = storage[wedge_ref].N[0];
  PointRef V_diff;
  while (next_ref != storage.NullSimplex()) {
    // Get a list of the vertices common to both simplices
    PointRef V_common[Traits::NDim];
    std::set_intersection(
        storage[prev_ref].V.begin(), storage[prev_ref].V.end(),
        storage[next_ref].V.begin(), storage[next_ref].V.end(), V_common);

    // Find the vertex in that list which is not in the pivot edge
    std::set_difference(V_common, V_common + Traits::NDim, V_edge,
                        V_edge + Traits::NDim - 1, &V_diff);

    // Walk across that vertex
    prev_ref = next_ref;
    next_ref = storage[prev_ref].NeighborAcross(V_diff);
  }

  SimplexRef neighbor_ref = prev_ref;
  storage[wedge_ref].N[i] = neighbor_ref;
  storage[neighbor_ref].SetNeighborAcross(V_diff, wedge_ref);
}

template <class Traits>
typename Traits::SimplexRef InsertOutside(
    typename Traits::Storage& storage, typename Traits::SimplexRef simplex_ref,
    typename Traits::PointRef point_ref) {
  typedef typename Traits::SimplexRef SimplexRef;
  typedef Facet<Traits> Facet;

  SimplexRef return_ref = storage.NullSimplex();

  std::list<SimplexRef> visible_hull;
  GetVisibleHull<Traits>(storage, simplex_ref, storage[point_ref],
                         std::back_inserter(visible_hull));

  // First, mark all members of the visible hull
  for (SimplexRef s_ref : visible_hull) {
    assert(storage[s_ref].V[0] == storage.NullPoint());
    storage[s_ref].marked[simplex::VISIBLE_HULL] = true;
    return_ref = s_ref;
  }

  // Replace infinite vertex with the newly added vertex
  DemoteHullSimplices<Traits>(storage, visible_hull, point_ref);

  // Build the horizon ridge, which is the set of all *edges* which border
  // the x-visible hull.
  auto horizon_ridge = BuildHorizonRidge<Traits>(storage, visible_hull);

  // Each edge of the horizon ridge emits a new simplex to fill the wedge
  // vacated by that facet of the demoted simplex.
  auto horizon_wedges =
      FillHorizonWedges<Traits>(storage, horizon_ridge, point_ref);

  // Now that the vacant wedges along the horizon ridge have been filled, we
  // must compute the neighborhood of these new simplices. The algorithm for
  // this is based on the horizon-ridge algorithm from Clarkson93. For each
  // new simplex along the horizon ridge that we need to find the neighbor of,
  // the neighbor that we need will lie across a *facet* of that simplex. That
  // facet is composed of N vertices, one of which is the peak vertex (the
  // newly added point). If we remove the peak vertex we are left with (N-1)
  // vertices which we will call the "pivot edge". We will do a walk around
  // this pivot edge until we find the neighboring simplex.

  // From our starting simplex we move to the neighbor across the infinite
  // vertex. This neighbor shares the pivot edge, and one other vertex in
  // common with the starting simplex. Incidentally, it is also a recently
  // demoted simplex. We continue the walk by following that
  // other vertex. A proof is given in the Clarkson93 paper that the walk
  // *must* end at the desired neighbor of the starting simplex.
  for (SimplexRef wedge_ref : horizon_wedges) {
    // For the i'th neighbor of the wedge simplex
    for (int i = 1; i < Traits::NDim + 1; i++) {
      // if this neighbor is already set, then just skip it
      if (storage[wedge_ref].N[i] != storage.NullSimplex()) {
        continue;
      }

      FindWedgeNeighbor<Traits>(storage, wedge_ref, i);
    }
  }

  /* Asci art for the link of a hull point in 2d
   *
   *     x ______
   *      /\    /
   *     /  \  /<--
   *    /____\/__
   *      A  /\
   *      | /  \
   *
   * The two facets with arrows pointing at them are the "link" of the point
   * x (on the hull boundary). The link is composed of all facets across from
   * the peak (inserted) vertex in all of the simplices that were demoted.
   */
  // stack of link facets
  std::list<Facet> link_facets;

  // push each facet of the link onto the link stack and clear it's mark
  for (SimplexRef s_ref : visible_hull) {
    storage[s_ref].marked[simplex::VISIBLE_HULL] = false;
    link_facets.emplace_back();
    link_facets.back().Construct(storage, s_ref,
                                 storage[s_ref].NeighborAcross(point_ref));
  }

  return Maintain(storage, point_ref, return_ref, link_facets);
}

template <class Traits>
typename Traits::SimplexRef InsertInside(
    typename Traits::Storage& storage, typename Traits::SimplexRef simplex_ref,
    typename Traits::PointRef point_ref) {
  typedef typename Traits::Simplex Simplex;
  typedef typename Traits::SimplexRef SimplexRef;
  typedef Facet<Traits> Facet;

  SimplexRef return_ref = storage.NullSimplex();

  // The insertion operation is equivalent to a facet-less "flip" so we will
  // reuse the code from InducedSubcomplex to handle insertions
  InducedSubcomplex<Traits> induced_subcomplex;

  // create vertex and neighbor set with point_ref included
  /* scope of i */ {
    Simplex& simplex = storage[simplex_ref];

    int i = 0;
    for (; i < Traits::NDim + 1 && simplex.V[i] < point_ref; ++i) {
      induced_subcomplex.V[i] = simplex.V[i];
      induced_subcomplex.S[i] = storage.NullSimplex();
    }
    induced_subcomplex.V[i] = point_ref;
    induced_subcomplex.S[i] = simplex_ref;
    ++i;
    for (; i < Traits::NDim + 2; ++i) {
      induced_subcomplex.V[i] = simplex.V[i - 1];
      induced_subcomplex.S[i] = storage.NullSimplex();
    }
  }

  // we are abusing the data structure here, but we do not need to Init or
  // Build because we have appropriately filled V and N above
  induced_subcomplex.Flip(storage);
  for (SimplexRef s_ref : induced_subcomplex.S) {
    if (s_ref != storage.NullSimplex()) {
      return_ref = s_ref;
    }
  }

  // Note: after the flip the original simplex will have been retired, so
  // simplex_ref is no longer valid

  /* Asci art for the link of a point in 2d
   *          |
   *       ___V__
   *      /\    /|
   * --->/  \  / |
   *    /____\/x |<-----
   *    \    /\  |
   * --->\  /  \ |
   *      \/____\|
   *          A
   *          |
   *
   * The six facets with arrows pointing at them are the "link" of the point
   * x (in the center).
   */
  // stack of link facets
  std::list<Facet> link_facets;

  // push each facet of the link onto the link stack
  for (int i = 0; i < Traits::NDim + 2; i++) {
    SimplexRef link_simplex = induced_subcomplex.S[i];
    if (link_simplex == storage.NullSimplex()) {
      continue;
    }
    SimplexRef neighbor = storage[link_simplex].NeighborAcross(point_ref);
    // skip any link facets which are on the hull of the point set
    if (storage[neighbor].V[0] == storage.NullPoint()) {
      continue;
    }
    link_facets.emplace_back();
    link_facets.back().Construct(storage, link_simplex, neighbor);
  }

  return Maintain(storage, point_ref, return_ref, link_facets);
}

/// Find the neighbor of a fill simplex that is across from v_ref
template <class Traits>
void FindFillNeighbor(typename Traits::Storage& storage,
                      typename Traits::SimplexRef s_ref,
                      typename Traits::PointRef v_ref,
                      typename Traits::PointRef peak_ref) {
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::PointRef PointRef;

  // Copy all but the 0'th and i'th vertices
  std::array<PointRef, Traits::NDim - 1> V_edge;
  std::array<PointRef, 2> V_missing = {v_ref, peak_ref};
  std::sort(V_missing.begin(), V_missing.end());
  std::set_difference(storage[s_ref].V.begin(), storage[s_ref].V.end(),
                      V_missing.begin(), V_missing.end(), V_edge.begin());

  SimplexRef prev_ref = s_ref;
  SimplexRef next_ref = storage[s_ref].NeighborAcross(peak_ref);
  PointRef V_diff;
  while (next_ref != storage.NullSimplex()) {
    // Get a list of the vertices common to both simplices
    std::array<PointRef, Traits::NDim> V_common;
    std::set_intersection(storage[prev_ref].V.begin(),
                          storage[prev_ref].V.end(),
                          storage[next_ref].V.begin(),
                          storage[next_ref].V.end(), V_common.begin());

    // Find the vertex in that list which is not in the pivot edge
    std::set_difference(V_common.begin(), V_common.end(), V_edge.begin(),
                        V_edge.end(), &V_diff);

    // Walk across that vertex
    prev_ref = next_ref;
    next_ref = storage[prev_ref].NeighborAcross(V_diff);
  }

  SimplexRef neighbor_ref = prev_ref;
  storage[s_ref].SetNeighborAcross(v_ref, neighbor_ref);
  storage[neighbor_ref].SetNeighborAcross(V_diff, s_ref);
}

template <class Traits, class Iterator>
typename Traits::SimplexRef InsertReplace(
    typename Traits::Storage& storage,
    typename Traits::PointRef point_ref,
    Iterator S_begin, Iterator S_end) {
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::PointRef PointRef;
  typedef Facet<Traits> Facet;

  SimplexRef return_ref = storage.NullSimplex();

  // first, mark all to-be-removed simplices
  for(Iterator s_iter = S_begin; s_iter != S_end; ++s_iter) {
    storage[*s_iter].marked[simplex::REMOVED_FOR_INSERT] = true;
  }

  std::list<SimplexRef> new_simplices;
  // for each simplex to be removed, find all neighbors which are not
  // to be removed
  for (Iterator s_iter = S_begin; s_iter != S_end; ++s_iter) {
    SimplexRef s_ref = *s_iter;
    for(SimplexRef neighbor_ref : storage[s_ref].N) {
      if(storage[neighbor_ref].marked[simplex::REMOVED_FOR_INSERT]) {
        continue;
      }

      // find the common facet
      std::array<PointRef, Traits::NDim + 1> V;
      PointRef v_in_removed;
      PointRef v_in_neighbor;
      set::IntersectionAndDifference(
          storage[s_ref].V.begin(), storage[s_ref].V.end(),
          storage[neighbor_ref].V.begin(), storage[neighbor_ref].V.end(),
          &v_in_removed, &v_in_neighbor, V.begin());
      // add the new point
      V.back() = point_ref;

      // create a new simplex
      SimplexRef new_ref = storage.Promote();
      new_simplices.push_back(new_ref);
      std::sort(V.begin(), V.end());
      std::copy(V.begin(), V.end(), storage[new_ref].V.begin());
      storage[new_ref].ComputeCenter(storage);

      // reset neighbors
      std::fill(storage[new_ref].N.begin(), storage[new_ref].N.end(),
                storage.NullSimplex());

      // assign the one neighbor relation that we know at this point
      storage[new_ref].SetNeighborAcross(point_ref, neighbor_ref);
      storage[neighbor_ref].SetNeighborAcross(v_in_neighbor, new_ref);
    }
  }

  // Now that the vacant wedges have been filled, we must compute the
  // neighborhood of these new simplices. The algorithm for
  // this is based on the horizon-ridge algorithm from Clarkson93. For each
  // new simplex that we need to find the neighbor of, the neighbor that we
  // need will lie across a *facet* of that simplex. That
  // facet is composed of N vertices, one of which is the peak vertex (the
  // newly added point). If we remove the peak vertex we are left with (N-1)
  // vertices which we will call the "pivot edge". We will do a walk around
  // this pivot edge until we find the neighboring simplex.

  // From our starting simplex we move to the neighbor across the peak
  // vertex. This neighbor shares the pivot edge, and one other vertex in
  // common with the starting simplex. We continue the walk by following that
  // other vertex. A proof is given in the Clarkson93 paper that the walk
  // *must* end at the desired neighbor of the starting simplex.
  for (SimplexRef wedge_ref : new_simplices) {
    // For the i'th neighbor of the wedge simplex
    for (PointRef v_ref : storage[wedge_ref].V) {
      // if this neighbor is already set, then just skip it
      if (storage[wedge_ref].NeighborAcross(v_ref) != storage.NullSimplex()) {
        continue;
      }

      FindFillNeighbor<Traits>(storage, wedge_ref, v_ref, point_ref);
    }
  }

  // Retire removed simplices
  for (Iterator s_iter = S_begin; s_iter != S_end; ++s_iter) {
    SimplexRef s_ref = *s_iter;
    storage[s_ref].marked[simplex::REMOVED_FOR_INSERT] = false;
    storage.Retire(s_ref);
  }

  // push each non-hull facet of the link onto the link stack and clear
  // marks
  std::list<Facet> link_facets;
  for (SimplexRef s_ref : new_simplices) {
    if (storage[s_ref].V[0] != storage.NullPoint()) {
      SimplexRef n_ref = storage[s_ref].NeighborAcross(point_ref);
      if (storage[n_ref].V[0] != storage.NullPoint()) {
        link_facets.emplace_back();
        link_facets.back().Construct(storage, s_ref, n_ref);
      }
    }
  }

  return_ref = new_simplices.front();
  return Maintain(storage, point_ref, return_ref, link_facets);
}

template <typename Traits>
typename Traits::SimplexRef FuzzyWalkInsert(
    typename Traits::Storage& storage, const typename Traits::SimplexRef s_0,
    const typename Traits::PointRef x_ref,
    const typename Traits::Scalar epsilon) {
  typedef typename Traits::SimplexRef SimplexRef;
  std::list<SimplexRef> interference_set;
  FuzzyWalk<Traits>(storage, s_0, storage[x_ref], epsilon,
                    std::back_inserter(interference_set));
  return InsertReplace<Traits>(storage, x_ref, interference_set.begin(),
                               interference_set.end());
}

}  // namespace edelsbrunner

#endif // EDELSBRUNNER96_TRIANGULATION_HPP_
