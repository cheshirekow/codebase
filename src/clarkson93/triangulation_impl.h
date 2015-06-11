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
#ifndef CLARKSON93_TRIANGULATION_IMPL_H_
#define CLARKSON93_TRIANGULATION_IMPL_H_

#include <cassert>
#include <algorithm>
#include <type_traits>

namespace clarkson93 {

template <class Traits>
Triangulation<Traits>::Triangulation() {
  clear();
}

template <class Traits>
Triangulation<Traits>::~Triangulation() {
  clear();
}

template <class Traits>
template <class Iterator, class Deiter>
void Triangulation<Traits>::init(Iterator begin, Iterator end, Deiter deiter) {
  // just some convenient storage for the initial simplices
  SimplexRef S[NDim + 1];

  // then, if we have enough for the first simplex, we need to build it
  int i = 0;
  SimplexRef S_0_ref = m_sMgr.create();
  Simplex& S_0 = m_deref.simplex(S_0_ref);

  for (Iterator ipRef = begin; ipRef != end && i < NDim + 1; ++ipRef)
    setVertex(S_0, i++) = deiter(ipRef);
  assert(i == NDim + 1);

  computeBase(S_0, m_deref);
  orientBase(S_0, m_deref.point(peak(S_0)), INSIDE);

  // we also need to setup all of the infinite simplices
  for (unsigned int i = 0; i < NDim + 1; i++) {
    S[i] = m_sMgr.create();
    Simplex& S_i = m_deref.simplex(S[i]);

    setVertex(S_i, 0) = m_antiOrigin;
    setNeighbor(S_i, 0) = S_0_ref;
    setNeighbor(S_0, i) = S[i];

    for (unsigned int j = 0; j < NDim + 1; j++) {
      int k = j;
      if (j < i)
        k++;
      else if (j == i)
        continue;

      setVertex(S_i, k) = vertex(S_0, j);
    }

    // we can't use the peak vertex of the inifinite simplices to
    // orient them, because it is fictious, so we'll orient it toward
    // the remaining point of the origin simplex
    computeBase(S_i, m_deref);
    orientBase(S_i, m_deref.point(vertex(S_0, i)), OUTSIDE);
  }

  // now we need to assign neighbors
  for (unsigned int i = 0; i < NDim + 1; i++) {
    for (unsigned int j = 1; j < NDim + 1; j++) {
      int k = (j <= i) ? j - 1 : j;
      Simplex& S_i = m_deref.simplex(S[i]);
      setNeighbor(S_i, j) = S[k];
    }
  }

  m_origin = S_0_ref;
  m_hullSimplex = S[0];

  // callback all of the initial hull faces
  for (unsigned int i = 0; i < NDim + 1; i++) m_callback.hullFaceAdded(S[i]);

  finish(S_0);
  for (unsigned int i = 0; i < NDim + 1; i++) {
    Simplex& S_i = m_deref.simplex(S[i]);
    finish(S_i);
    setMember(S_i, simplex::HULL) = true;
  }
}

template <class Traits>
void Triangulation<Traits>::insert(PointRef x, SimplexRef S) {
  SimplexRef So_ref = find_x_visible(x, S);
  Simplex& So = m_deref.simplex(So_ref);
  if (!isMember(So, simplex::HULL)) return;

  fill_x_visible(s_optLvl, x, So_ref);
  alter_x_visible(s_optLvl, x);
}

template <class Traits>
void Triangulation<Traits>::insert(PointRef x) {
  insert(x, m_origin);
}

template <class Traits>
void Triangulation<Traits>::clear() {
  m_hullSimplex = 0;
  m_origin = 0;
  m_sMgr.clear();
  m_xv_queue.clear();
  m_xv_walked.clear();
  m_xvh.clear();
  m_xvh_queue.clear();
  m_ridges.clear();
}

template <class Traits>
typename Traits::Simplex* Triangulation<Traits>::find_x_visible(
    PointRef Xref, SimplexRef Sref) {
  // turn generic reference into a real reference
  Point& x = m_deref.point(Xref);

  // first clear out our search structures
  // so the flags get reset, we do this at the beginning so that after the
  // the update they persist and we can view them for debugging
  for (SimplexRef Sref : m_xv_walked) {
    Simplex& S = m_deref.simplex(Sref);
    setMember(S, simplex::XVISIBLE_WALK) = false;
  }
  m_xv_queue.clear();
  m_xv_walked.clear();

  // if the origin simplex is not visible then start at the neighbor
  // across his base, which must be x-visible
  {
    Simplex& S = m_deref.simplex(Sref);

    if (!isVisible(S, x)) Sref = neighborAcross(S, peak(S));
  }

  // is set to true when we find an infinite x-visible simplex
  bool foundVisibleHull = false;

  // reference scope
  {
    Simplex& S = m_deref.simplex(Sref);

    // sanity check
    assert(isVisible(S, x));

    // start the search at S
    m_xv_queue.push(PQ_Key(0, Sref));
    m_xv_walked.push_back(Sref);

    setMember(S, simplex::XVISIBLE_WALK) = true;

    // if the seed we've been given is both x-visible and infinite
    // then there is no need for a walk
    foundVisibleHull = isInfinite(S, m_antiOrigin);
  }

  // starting at the given simplex, walk in the direction of x until
  // we find an x-visible infinite simplex (i.e. hull facet)
  while (m_xv_queue.size() > 0 && !foundVisibleHull) {
    SimplexRef pop_ref = m_xv_queue.pop().val;
    Simplex& pop = m_deref.simplex(pop_ref);

    // the neighbor we skip b/c it was how we were found
    SimplexRef parent = neighborAcross(pop, peak(pop));

    // for each neighbor except the one across the base facet, queue all
    // x-visible simplices sorted by base facet distance to x
    for (SimplexRef Nref : neighborhood(pop)) {
      if (Nref == parent) continue;

      Simplex& N = m_deref.simplex(Nref);

      // if the neighbor is x-visible but has not already been queued or
      // expanded, then add it to the queue
      if (isVisible(N, x) && !isMember(N, simplex::XVISIBLE_WALK)) {
        // if the base facet is x-visible and the simplex is also
        // infinite then we have found our x-visible hull facet
        if (isInfinite(N, m_antiOrigin)) {
          Sref = Nref;
          foundVisibleHull = true;
          break;
        }

        // otherwise add N to the search queue
        Scalar d = (x - m_deref.point(peak(N))).squaredNorm();
        // Scalar d = normalProjection( N, x );

        m_xv_walked.push_back(Nref);
        m_xv_queue.push(PQ_Key(d, Nref));

        setMember(N, simplex::XVISIBLE_WALK) = true;
      }
    }
  }

  // if we didn't find a hull facet then the point is inside the triangulation
  // and is redundant, so lets just give it up
  return Sref;
}

template <class Traits>
void Triangulation<Traits>::fill_x_visible(const OptLevel<0>&, PointRef Xref,
                                           SimplexRef Sref) {
  Point& x = m_deref.point(Xref);
  for (SimplexRef S_ref : m_xvh) {
    Simplex& S = m_deref.simplex(S_ref);
    setMember(S, simplex::XV_HULL) = false;
  }

  for (Ridge& ridge : m_ridges) {
    Simplex& S = m_deref.simplex(ridge.Sfill);
    setMember(S, simplex::HORIZON_FILL) = false;
  }

  m_xvh.clear();
  m_xvh_queue.clear();
  m_ridges.clear();

  // this S is both x-visible and infinite, so we do an expansive
  // search for all such simplicies around it. So we initialize the search
  // stack with this simplex
  {
    m_xvh.push_back(Sref);
    m_xvh_queue.push_back(Sref);

    Simplex& S = m_deref.simplex(Sref);
    setMember(S, simplex::XV_HULL) = true;
  }

  // at each iteration...
  while (m_xvh_queue.size() > 0) {
    // pop one simplex off the stack
    SimplexRef Sref = m_xvh_queue.back();
    Simplex& S = m_deref.simplex(Sref);
    m_xvh_queue.pop_back();

    // and check all of it's infinite neighbors
    for (SimplexRef Nref : neighborhood(S)) {
      // skip the finite neighbor
      if (Nref == peakNeighbor(S)) continue;

      Simplex& N = m_deref.simplex(Nref);
      bool xVisible = isVisible(N, x);

      // sanity check
      assert(isInfinite(N, m_antiOrigin));

      // if the neighbor is both infinite and x-visible , but has not
      // yet been queued for expansion, then add it to
      // the x-visible hull set, and queue it up for expansion
      if (xVisible && !isMember(N, simplex::XV_HULL)) {
        setMember(N, simplex::XV_HULL) = true;
        m_xvh_queue.push_back(Nref);
        m_xvh.push_back(Nref);
      }

      // if is not visible then there is a horizon ridge between this
      // simplex and the neighbor simplex
      if (!xVisible) m_ridges.emplace_back(Sref, Nref);
    }
  }
}

template <class Traits>
void Triangulation<Traits>::alter_x_visible(const OptLevel<0>&, PointRef Xref) {
  // first we go through all the x-visible simplices, and replace their
  // peak vertex (the ficitious anti-origin) with the new point x, and then
  // notify any hooks that the simplex was removed from the hull
  for (SimplexRef Sref : m_xvh) {
    Simplex& S = m_deref.simplex(Sref);
    setPeak(S) = Xref;
    finish(S);
    setMember(S, simplex::HULL) = false;
    m_callback.hullFaceRemoved(Sref);
  }

  // now for each horizon ridge we have to construct a new simplex
  for (Ridge& ridge : m_ridges) {
    // allocate a new simplex and add it to the list
    SimplexRef Snew = m_sMgr.create();
    ridge.Sfill = Snew;

    // in order to traverse the hull we need at least one hull simplex and
    // since all new simplices are hull simlices, we can set one here
    m_hullSimplex = Snew;

    // In the parlance of Clarkson, we have two simplices V and N
    // note that V is ridge->Svis and N is ridge->Sinvis
    // V was an infinite simplex and became
    // a finite one... and N is still an infinite simplex
    Simplex& V = m_deref.simplex(ridge.Svis);
    Simplex& N = m_deref.simplex(ridge.Sinvis);
    Simplex& S = m_deref.simplex(ridge.Sfill);
    setMember(S, simplex::HULL) = true;

    // the new simplex has a peak vertex at the anti-origin, across from
    // that vertex is V.
    setVertex(S, 0) = m_antiOrigin;
    setNeighbor(S, 0) = ridge.Svis;

    // the new simplex also contains x as a vertex and across from that
    // vertex is N
    setVertex(S, 1) = Xref;
    setNeighbor(S, 1) = ridge.Sinvis;

    // split the vertex set of V and N into those that are only in V,
    // those that are only in N, and those that are common
    std::vector<PointRef> vRidge, vV, vN;
    vRidge.reserve(NDim - 1);
    vsetSplit(V, N, std::back_inserter(vV), std::back_inserter(vN),
              std::back_inserter(vRidge));

    // Note: when vsetSplit is finished vV should contain two vertices,
    // one of which is x, and vN should contain two vertices, one of which
    // is the anti origin, and vRidge should contain NDim-2 vertices
    assert(vV.size() == 2);
    assert(vN.size() == 2);
    assert(vRidge.size() == NDim - 1);

    int i = 2;
    for (PointRef v : vRidge) setVertex(S, i++) = v;

    // we only care about the vertex which is not x or the anti origin
    // so make sure we can identify that point
    if (vV.back() != Xref) vV.front() = vV.back();
    if (vN.back() != m_antiOrigin) vN.front() = vN.back();

    // now we can assign these neighbors correctly
    setNeighborAcross(V, vV[0]) = Snew;
    setNeighborAcross(N, vN[0]) = Snew;

    // the neighbors will continue to be null, but we can sort the
    // vertices now
    finish(S);

    // mark this simplex as a member of the horizon wedge fill
    setMember(S, simplex::HORIZON_FILL) = true;

    // we'll need to use the half-space inequality at some point but
    // we can't calcualte it normally b/c one of the vertices isn't at
    // a real location, so we calculate it to be coincident to the base
    // facet, and then orient it so that the vertex of V which is not
    // part of the new simplex is on the other side of the constraint
    computeBase(S, m_deref);
    orientBase(S, m_deref.point(vV[0]), OUTSIDE);
  }

  // ok now that all the new simplices have been added, we need to go
  // and assign neighbors to these new simplicies
  for (Ridge& ridge : m_ridges) {
    Simplex& S = m_deref.simplex(ridge.Sfill);
    Simplex& V = m_deref.simplex(ridge.Svis);
    Simplex& N = m_deref.simplex(ridge.Sinvis);

    // let x be a vertex of S for which a neighbor has not been assigned,
    // then the facet f across from x is a set of NDim vertices. Let
    // f_N be the (NDim-1) vertices f \ {antiOrigin}. Well then V must
    // share that facet. Furthermore, if we step into V then V only has
    // one neighbor other than S which shares that facet. Each simplex
    // that shares this facet has exactly two neighbors sharing that
    // facet. The missing neighbor of S also shares that facet. Thus we
    // walk through the simplices S->V->...->N_x until we find N_x which
    // will be the first HORIZON_FILL simplex on that path.

    // so start by building the horizon ridge wedge facet as a set of
    // vertices
    std::set<PointRef> ridgeFacet;
    vsetIntersection(V, N, std::inserter(ridgeFacet, ridgeFacet.begin()));
    assert(ridgeFacet.size() == NDim - 1);

    // now for each vertex in that set, we need to find the neighbor
    // across from that vertex
    for (PointRef v : ridgeFacet) {
      // so build the edge which is the ridge facet without that
      // particular vertex
      std::vector<PointRef> edge;
      edge.reserve(NDim - 1);
      std::copy_if(ridgeFacet.begin(), ridgeFacet.end(),
                   std::back_inserter(edge),
                   [v](PointRef q) { return q != v; });
      edge.push_back(Xref);
      assert(edge.size() == NDim - 1);

      // now start our walk with S and V
      setMember(S, simplex::S_WALK) = true;
      setMember(V, simplex::S_WALK) = true;

      // create a search queue (will only ever hold 2 elements)
      std::vector<SimplexRef> queue;
      std::vector<SimplexRef> expanded;
      std::vector<SimplexRef> neighborhood;
      queue.reserve(2);
      queue.push_back(ridge.Svis);
      expanded.push_back(ridge.Svis);
      expanded.push_back(ridge.Sfill);

      // this is where we'll put the found neighbor
      SimplexRef Nfound;
      bool foundNeighbor = false;

      // while the queue is not empty
      while (queue.size() > 0 && !foundNeighbor) {
        // pop one element off the queue
        SimplexRef Sc_ref = queue.back();
        queue.pop_back();
        Simplex& Sc = m_deref.simplex(Sc_ref);

        // find all the neighbors that share the facet
        neighborhood.clear();
        neighborSharing(Sc, edge.begin(), edge.end(),
                        std::back_inserter(neighborhood));
        assert(neighborhood.size() == 2);

        // get the one neighbor which isn't already in S_WALK
        for (SimplexRef Nref : neighborhood) {
          Simplex& N = m_deref.simplex(Nref);

          // if it's not the one we came from
          if (!isMember(N, simplex::S_WALK)) {
            // if N is in HORIZON_FILL then this is the neighbor
            // we are looking for
            if (isMember(N, simplex::HORIZON_FILL)) {
              Nfound = Nref;
              foundNeighbor = true;
              break;
            }

            queue.push_back(Nref);
            expanded.push_back(Nref);
            setMember(N, simplex::S_WALK) = true;
          }
        }

        // sanity check
        assert(queue.size() < 2);
      }

      for (SimplexRef Ss : expanded)
        setMember(m_deref.simplex(Ss), simplex::S_WALK) = false;

      // sanity check
      assert(foundNeighbor);
      assert(Nfound != ridge.Sfill);

      // set the other guy as our neighbor, we dont do the inverse
      // setting of the neighbor as it will happen when Nfound is
      // processedlater, note that this incurs a doubling of the
      // amount of S_WALKing that we do and can possibly be optimized
      setNeighborAcross(S, v) = Nfound;
    }
  }

  // now that neighbors have been assigned we can inform any listeners
  for (Ridge& ridge : m_ridges) m_callback.hullFaceAdded(ridge.Sfill);
}

}  // namespace clarkson93

#endif  // CLARKSON93_TRIANGULATION_IMPL_H_
