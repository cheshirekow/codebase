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
#ifndef CLARKSON93_TRIANGULATION_IMPL_H_
#define CLARKSON93_TRIANGULATION_IMPL_H_

#include <clarkson93/triangulation.h>

#include <array>
#include <cassert>
#include <algorithm>
#include <type_traits>

#include <clarkson93/bit_member.h>
#include <clarkson93/simplex_impl.h>

namespace clarkson93 {

template <class Traits>
Triangulation<Traits>::Triangulation(PointRef anti_origin,
                                     SimplexAllocator* alloc)
    : anti_origin_(anti_origin), alloc_(alloc) {
  Clear();
}

template <class Traits>
Triangulation<Traits>::~Triangulation() {
  Clear();
}

template <class Traits>
template <class Container>
void Triangulation<Traits>::BuildInitial(const Container& vertices,
                                         const Deref& deref) {
  // just some convenient storage for the initial simplices
  std::array<Simplex<Traits>*, kDim + 1> S;

  int i = 0;
  Simplex<Traits>* s0_ptr = alloc_->Create();
  Simplex<Traits>& s0 = *s0_ptr;

  // fill the initial simplex with the vertex set
  assert(vertices.size() > kDim + 1);
  auto vertex_iterator = s0.V.begin();
  for (auto vertex_id : vertices) {
    *vertex_iterator++ = vertex_id;
    if (vertex_iterator == s0.V.end()) {
      break;
    }
  }
  assert(vertex_iterator == s0.V.end());

  ComputeBase(s0, deref);
  OrientBase(s0, deref(s0.GetPeakVertex()), INSIDE);

  // construct and initialize infinite simplices
  for (int i = 0; i < kDim + 1; i++) {
    S[i] = alloc_->Create();
    Simplex& s_i = *S[i];

    // the neighbor across the anti-origin is the origin simplex, and
    // this simplex is the i'th neighbor of the origin simplex
    s_i.V[0] = anti_origin_simplex_;
    s_i.N[0] = s0_ptr;
    s0.N[i] = S[i];

    for (int j = 0; j < kDim + 1; j++) {
      if (j == i)
        continue;

      const int k = j < i ? j + 1 : j;
      s_i.V[k] = s0.V[j];
    }

    // we can't use the peak vertex of the inifinite simplices to
    // orient them, because it is fictious, so we'll orient it toward
    // the remaining point of the origin simplex
    ComputeBase(S[i], deref);
    OrientBase(S[i], deref(s0.V[i]), simplex::OUTSIDE);
  }

  // now we need to assign mutually inifinite neighbors
  for (int i = 0; i < kDim + 1; i++) {
    for (int j = 1; j < kDim + 1; j++) {
      const int k = (j <= i) ? j - 1 : j;
      S[i]->N[j] = S[k];
    }
  }

  origin_simplex_ = s0_ptr;
  hull_simplex_ = S[0];

  // callback all of the initial hull faces
  //  for (int i = 0; i < kDim + 1; i++)
  //    m_callback.hullFaceAdded(S[i]);

  // Finalize the simplices by sorting their vertex sets
  SortVertices(s0_ptr);
  for (int i = 0; i < kDim + 1; i++) {
    SortVertices(S[i]);
    S_i->AddTo(simplex::HULL);
  }

  // remember which simplices we created
  allocated_simplices_.emplace(s0_ptr);
  std::copy(S.begin(), S.end(), std::back_inserter(allocated_simplices_));
}

template <class Traits>
void Triangulation<Traits>::Insert(PointRef vertex_id, const Deref& deref,
                                   Simplex<Traits>* search_start) {
  assert(origin_simplex_);
  Simplex<Traits>* s0_ptr = FindXVisible(vertex_id, search_start);
  // if the inserted point is not outside the current hull then we do nothing
  if (!s0_ptr->IsMemberOf(simplex::HULL)) {
    return;
  }

  FillXVisible(vertex_id, s0_ptr);
  AlterXVisible(vertex_id);
}

template <class Traits>
void Triangulation<Traits>::Insert(PointRef vertex_id, const Deref& deref) {
  assert(origin_simplex_);
  Insert(vertex_id, origin_simplex_);
}

template <class Traits>
void Triangulation<Traits>::Clear() {
  hull_simplex_ = nullptr;
  origin_simplex_ = nullptr;

  for (auto simplex_ptr : allocated_simplices_) {
    alloc_->Free(simplex_ptr);
  }
}

template <class Traits>
typename Simplex<Traits>* Triangulation<Traits>::FindVisibleHull(
    PointRef vertex_id, const Deref& deref, Simplex<Traits>* search_start) {
  // set of simplices that have been walked
  BitMemberSet<simplex::Sets> walked_set(simplex::VISIBLE_WALK);

  // turn generic reference into a real reference
  auto& vertex = deref.point(vertex_id);

  // first clear out our search structures
  // so the flags get reset, we do this at the beginning so that after the
  // the update they persist and we can view them for debugging
  for (Simplex<Traits>* s_ptr : xv_walked_) {
    walked_set.Remove(s_ptr);
  }
  xv_queue_.clear();
  xv_walked_.clear();

  // if the origin simplex is not visible then start at the neighbor
  // across his base, which must be x-visible
  if (!IsVisible(*search_start, vertex)) {
    search_start = search_start->GetPeakNeighbor();
  }
  // sanity check
  assert(IsVisible(*search_start, vertex));

  // is set to true when we find an infinite x-visible simplex
  // if the seed we've been given is both x-visible and infinite
  // then there is no need for a walk
  bool found_visible_hull = IsInfinite(search_start, anti_origin_);
  Simplex<Traits>* found_simplex = search_start;

  // start the search
  xv_queue_.Push({0, search_start});
  xv_walked_.push_back(search_start);
  walked_set.Add(search_start);

  // starting at the given simplex, walk in the direction of x until
  // we find an x-visible infinite simplex (i.e. hull facet)
  while (xv_queue_.size() > 0 && !found_visible_hull) {
    Simplex<Traits>* pop_ptr = xv_queue_.Pop().val;

    // the neighbor we skip b/c it was how we entered this simplex
    Simplex<Traits>* parent_ptr = pop_ptr->GetPeakNeighbor();

    // for each neighbor except the one across the base facet, queue all
    // x-visible simplices sorted by base facet distance to x
    for (Simplex<Traits>* neighbor_ptr : Neighborhood(*pop_ptr)) {
      if (neighbor_ptr == parent_ptr) {
        continue;
      }

      // if the neighbor is x-visible but has not already been queued or
      // expanded, then add it to the queue
      if (IsVisible(*neighbor_ptr, vertex) &&
          !walked_set.IsMember(neighbor_ptr)) {
        // if the base facet is x-visible and the simplex is also
        // infinite then we have found our x-visible hull facet
        if (IsInfinite(*neighbor_ptr, anti_origin_)) {
          found_simplex = neighbor_ptr;
          found_visible_hull = true;
          break;
        }

        // otherwise add the neighbor to the search queue
        Scalar d =
            (vertex - deref(neighbor_ptr->GetPeakVertex())).squaredNorm();
        // Scalar d = NormalProjection(*neighbor_ptr, vertex);

        xv_walked_.push_back(neighbor_ptr);
        xv_queue_.push({d, neighbor_ptr});
        walked_set.Add(neighbor_ptr);
      }
    }
  }

  // If we didn't find a hull facet then the point is inside the triangulation
  // and is redundant, so we'll just give up and return the search start as
  // the found simplex. The caller will know what this means.
  return found_simplex;
}

template <class Traits>
void Triangulation<Traits>::FloodVisibleHull(
    PointRef vertex_id, const Deref& deref,
    Simplex<Traits>* visible_hull_simplex) {
  // set of hull simplices that are visible
  BitMemberSet<simplex::Sets> visible_hull_set(simplex::VISIBLE_HULL);
  BitMemberSet<simplex::Sets> horizon_fill_set(simplex::HORIZON_FILL);

  auto& vertx = deref(vertex_id);

  // clear old results
  for (Simplex<Traits>* simplex_ptr : xvh_) {
    visible_hull_set.Remove(simplex_ptr);
  }

  for (Ridge& ridge : ridges_) {
    horizon_fill_set.Remove(ridge.fill);
  }

  xvh_.clear();
  ridges_.clear();

  // the first simplex  is both x-visible and infinite, so we do an expansive
  // search for all such simplicies around it. So we initialize the search
  // stack with this simplex
  xvh_.push_back(visible_hull_simplex);
  visible_hull_set.Add(visible_hull_simplex);

  // visible_hull_iter separates the visible hull list into those that have
  // been expanded (before iter) and those that have not (iter onwards),
  // allowing
  // us to do a breadth-first search with a single data structure
  auto visible_hull_iter = xvh_.begin();
  while (visible_hull_iter != xvh_.end()) {
    // pop one simplex off the stack
    Simplex<Traits>* pop_ptr = *visible_hull_iter++;

    // and check all of it's infinite neighbors
    for (Simplex<Traits>* neighbor_ptr : Neighborhood(*pop_ptr)) {
      // skip the finite neighbor, because it is finite
      if (neighbor_ptr == pop_ptr->GetPeakNeighbor()) {
        continue;
      }

      bool is_x_visible = IsVisible(*neighbor_ptr, vertex);

      // sanity check
      assert(IsInfinite(*neighbor_ptr, anti_origin_));

      // if the neighbor is both infinite and x-visible , but has not
      // yet been queued for expansion, then add it to
      // the x-visible hull set, and queue it up for expansion
      if (is_x_visible && !visible_hull_set.IsMember(*neighbor_ptr) {
        visible_hull_set.Add(neighbor_ptr);
        xvh_.push_back(neighbor_ptr);
      }

      // if is not visible then there is a horizon ridge between this
      // simplex and the neighbor simplex
      if (!is_x_visible)
        ridges_.emplace_back(pop_ref, neighbor_ref);
    }
  }
}

template <class Traits>
void Triangulation<Traits>::FillVisibleHull(PointRef vertex_id,
                                            const Deref& deref) {
  // set of hull simplices (we will remove some here)
  BitMemberSet<simplex::Sets> hull_set(simplex::HULL);

  // set of hull simplices that are visible
  BitMemberSet<simplex::Sets> visible_hull_set(simplex::VISIBLE_HULL);

  // set of hull simplices that are on the horizon ridge
  BitMemberSet<simplex::Sets> horizon_set(simplex::HORIZON);

  // set of hull simplices that were created during the fill operation
  BitMemberSet<simplex::Sets> horizon_fill_set(simplex::HORIZON_FILL);

  // set of simplices marked during a feature walk
  BitMemberSet<simplex::Sets> feature_walk_set(simplex::FEATURE_WALK);

  auto& vertex = deref(vertex_id);

  // first we go through all the x-visible simplices, and replace their
  // peak vertex (the ficitious anti-origin) with the new point x, and then
  // notify any hooks that the simplex was removed from the hull
  for (Simplex<Traits>* s_ptr : xvh_) {
    s_ptr->SetPeak(vertex_id);
    hull_set.Remove(s_ptr);
    //    m_callback.hullFaceRemoved(Sref);
  }

  // now for each horizon ridge we have to construct a new simplex
  for (Ridge& ridge : ridges_) {
    // allocate a new simplex and add it to the list
    Simplex<Traits>* new_ptr = alloc_->Create();
    allocated_simplices_.push_back(new_ptr);
    ridge.fill = new_ptr;
    hull_set.Add(new_ptr);

    // in order to traverse the hull we need at least one hull simplex and
    // since all new simplices are hull simlices, we can set one here
    hull_simplex_ = new_ptr;

    // In the parlance of Clarkson, we have two simplices V and N
    // note that V is ridge->x_visible and N is ridge->x_invisible
    // V was an infinite simplex and became
    // a finite one... and N is still an infinite simplex
    Simplex<Traits>& V = *ridge.x_visible;
    Simplex<Traits>& N = *ridge.x_invisible;
    Simplex<Traits>& S = *ridge.fill;

    // the new simplex has a peak vertex at the anti-origin, across from
    // that vertex is V.
    S.V[0] = anti_origin_;
    S.N[0] = ridge.x_visible;

    // the new simplex also contains x as a vertex and across from that
    // vertex is N
    S.V[1] = vertex_id;
    S.N[1] = ridge.x_invisible;

    // split the vertex set of V and N into those that are only in V,
    // those that are only in N, and those that are common
    std::array<PointRef, kDim - 1> ridge_vertices;
    std::array<PointRef, 2> V_vertices, N_vertices;
    VsetSplit(V, N, V_vertices.begin(), N_vertices.begin(),
              ridge_vertices.begin());

    // Note: when vsetSplit is finished V_vertices should contain two vertices,
    // one of which is x, and N_vertices should contain two vertices, one of
    // which
    // is the anti origin, and rige_vertices should contain kDim-2 vertices
    int i = 2;
    for (PointRef v : ridge_vertices) {
      S.V[i++] = v;
    }

    // we only care about the vertex which is not x or the anti origin
    // so make sure we can identify that point by putting it in the zero slot
    if (V_vertices.back() != vertex_id) {
      std::swap(V_vertices.front(), V_vertices.back());
    }
    if (N_vertices.back() != anti_origin_) {
      std::swap(N_vertices.front(), N_vertices.back());
    }

    // now we can assign these neighbors correctly
    V.SetNeighborAcross(V_vertices[0], new_ptr);
    N.SetNeighborAcross(N_vertices[0], new_ptr);

    // the neighbors will continue to be null, but we can sort the
    // vertices now
    SortVertices(ridge.fill);

    // mark this simplex as a member of the horizon wedge fill
    horizon_fill_set.Add(ridge.fill);

    // we'll need to use the half-space inequality at some point but
    // we can't calcualte it normally b/c one of the vertices isn't at
    // a real location, so we calculate it to be coincident to the base
    // facet, and then orient it so that the vertex of V which is not
    // part of the new simplex is on the other side of the constraint
    ComputeBase(ridge.fill, deref);
    OrientBase(ridge.fill, deref(V_vertices[0]), simplex::OUTSIDE);
  }

  // ok now that all the new simplices have been added, we need to go
  // and assign neighbors to these new simplicies
  for (Ridge& ridge : ridges_) {
    Simplex<Traits>& S = *ridge.fill;
    Simplex<Traits>& V = *ridge.x_visible;
    Simplex<Traits>& N = *ridge.x_invisible;

    // let x be a vertex of S for which a neighbor has not been assigned,
    // then the facet f across from x is a set of kDim vertices. Let
    // f_N be the (kDim-1) vertices f \ {antiOrigin}. Well then V must
    // share that facet. Furthermore, if we step into V then V only has
    // one neighbor other than S which shares that facet. Each simplex
    // that shares this facet has exactly two neighbors sharing that
    // facet. The missing neighbor of S also shares that facet. Thus we
    // walk through the simplices S->V->...->N_x until we find N_x which
    // will be the first HORIZON_FILL simplex on that path.

    // so start by building the horizon ridge wedge facet as a set of
    // vertices
    std::array<PointRef, kDim - 1> ridge_facet;
    VsetIntersection(V, N, ridge_facet.begin());

    // now for each vertex in that set, we need to find the neighbor
    // across from that vertex
    for (PointRef v : ridge_facet) {
      // so build the edge which is the ridge facet without that
      // particular vertex
      // TODO(josh) : this is basically set-minus, set-plus. It would be cool
      // to do have a template expression library for working with sorted arrays
      std::array<PointRef, kDim - 1> edge;
      std::copy_if(ridge_facet.begin(), ridge_facet.end(), edge.begin(),
                   [v](PointRef q) { return q != v; });
      edge.back() = vertex_id;
      std::sort(edge.begin(), edge.end());

      // now start our walk with S and V
      std::list<Simplex<Traits>*> feature_walk_list;
      feature_walk_list.push_back(ridge.fill);
      feature_walk_list.push_back(ridge.x_visible);
      feature_walk_set.Add(ridge.fill);
      feature_walk_set.Add(risge.x_visible);

      // create a search queue (will only ever hold 2 elements)
      std::list<Simplex<Traits>*> queue;
      queue.push_back(ridge.x_visible);

      // this is where we'll put the found neighbor
      Simplex<Traits>* found_neighbor = nullptr;

      // while the queue is not empty
      while (queue.size() > 0 && !found_neighbor) {
        // pop one element off the queue
        Simplex<Traits>* pop_ptr = queue.back();
        queue.pop_back();

        // find all the neighbors that share the facet
        assert(GetNeighborsSharing(*pop_ref, edge).size() == 2);

        // get the one neighbor which isn't already in FEATURE_WALK
        for (Simplex<Traits>* neighbor_ptr :
             GetNeighborsSharing(*pop_ref, edge)) {
          // if it's not the one we came from
          if (!feature_walk_set.IsMember(*neighbor_ptr)) {
            // if N is in HORIZON_FILL then this is the neighbor
            // we are looking for
            if (horizon_fill_set.IsMember(*neighbor_ptr)) {
              found_neighbor = neighbor_ptr;
              break;
            }

            queue.push_back(neighbor_ptr);
            feature_walk_list.push_back(neighbor_ptr);
            feature_walk_set.Add(neighbor_ptr);
          }
        }

        // sanity check
        assert(queue.size() < 2);
      }

      // we can clear the walk set
      for (Simplex<Traits>* s_ptr : feature_walk_list) {
        feature_walk_set.Remove(s_ptr);
      }

      // sanity check
      assert(found_neighbor);
      assert(found_neighbor != ridge.fill);

      // set the other guy as our neighbor, we dont do the inverse
      // setting of the neighbor as it will happen when Nfound is
      // processedlater, note that this incurs a doubling of the
      // amount of S_WALKing that we do and can possibly be optimized
      S.SetNeighborAcross(vertex_id) = found_neighbor;
    }
  }

  // now that neighbors have been assigned we can inform any listeners
  //  for (Ridge& ridge : m_ridges) {
  //    m_callback.hullFaceAdded(ridge.Sfill);
  //  }
}

}  // namespace clarkson93

#endif  // CLARKSON93_TRIANGULATION_IMPL_H_
