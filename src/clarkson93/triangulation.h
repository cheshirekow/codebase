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
#ifndef CLARKSON93_TRIANGULATION_H_
#define CLARKSON93_TRIANGULATION_H_

#include <initializer_list>
#include <list>
#include <queue>
#include <set>
#include <clarkson93/horizon_ridge.h>
#include <clarkson93/indexed.h>
#include <clarkson93/priority_queue.h>
#include <clarkson93/simplex.h>

namespace clarkson93 {

/// misleadingly-named data structure, is actually a "simplexification", the
/// dimension agnostic analog of a triangulation
/**
 *  By the convensions of Clarkson.compgeo93, the first (i.e. 0 index) simplex
 *  is the "origin" simplex. The 'origin' (@f$ O @f$) is a point inside that
 * simplex
 *  (we'll pick the geometric mean of the vertices). The 'anti-origin'
 *  (@f$ \overbar{o} @f$) is a fictitous vertex which is a member of all
 *  simplices
 *  on the convex hull. It is analogous to CGAL's 'vertex at infinity'. We'll
 *  give it a value of @f$ [0 0 0...] @f$, but it's address is used to identify
 *  it as a special point, and it's value is never used in calculations.
 */
template <class Traits>
class Triangulation {
 public:
  // Typedefs
  // -----------------------------------------------------------------------
  static const int kDim = Traits::kDim;

  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::PointRef PointRef;
  typedef typename Traits::Deref Deref;
  typedef typename Traits::SimplexAllocator SimplexAllocator;

  typedef Triangulation<Traits> This;
  typedef HorizonRidge<Traits> Ridge;

  typedef Indexed<Scalar, Simplex<Traits>*> QueueElement;
  typedef PriorityQueue<QueueElement> WalkQueue;

  typedef std::list<Simplex<Traits>*> SimplexList;
  typedef std::list<Ridge> HorizonList;

 public:
  // Data Members
  // -----------------------------------------------------------------------
  Simplex<Traits>* hull_simplex_;    ///< a simplex in the hull
  Simplex<Traits>* origin_simplex_;  ///< origin simplex
  PointRef anti_origin_;             ///< fictitious point
  SimplexAllocator* alloc_;          ///< allocator for simplices

  WalkQueue xv_queue_;     ///< walk for x-visible search
  SimplexList xv_walked_;  ///< set of simplices ever expanded for
                           ///  the walk
  SimplexList xvh_;        ///< set of x-visible hull simplices

  HorizonList ridges_;  ///< set of horizon ridges
  SimplexList allocated_simplices_;

 public:
  /// anti_origin must have a value less than all vertex id's that are expected
  /// to be used.
  Triangulation(PointRef anti_origin, SimplexAllocator* alloc);
  ~Triangulation();

  /// builds the initial triangulation from the first @p NDim + 1 points
  /// inserted
  template <class Container>
  void BuildInitial(const Container& vertices, const Deref& deref);

  template <typename T>
  void BuildFromIL(const std::initializer_list<T>& vertices,
                   const Deref& deref) {
    BuildInitial(vertices, deref);
  }

  /// insert a new point into the triangulation and update the convex
  /// hull (if necessary)
  /**
   *  @param  x   const ref to the new point to add
   *  @param  S   an x-visible facet, if null (default) will search for one
   */
  void Insert(PointRef vertex_id, const Deref& deref,
              Simplex<Traits>* search_start);

  void Insert(PointRef vertex_id, const Deref& deref);

  /// destroys all simplex objects that have been generated and clears all
  /// lists/ sets
  void Clear();

  /// find the set of @f$ x\mathrm{-visible} @f$ facets and put them in
  /// @p x_visible
  /**
   *  Using the first method of Clarkson.compgeo93, we walk along the segment
   *  @f$ \overbar{Ox} @f$, starting at @f$ O @f$ . If this walk enters a
   *  simplex whose peak vertex is the anti-origin, then
   *  an x-visible current facet has been found. Otherwise a simplex of
   *  T containing @f$ x @f$ has been found, showing that
   *  @f$ x \in \mathrm{hull} R @f$
   */
  Simplex<Traits>* FindVisibleHull(PointRef vertex_id, const Deref& deref,
                                   Simplex<Traits>* search_start);

  /// given a simplex S which is x-visible and infinite, fill the set of
  /// all x-visible and infinite facets
  void FloodVisibleHull(PointRef vertex_id, const Deref& deref,
                        Simplex<Traits>* visible_hull_simplex);

  /// update each x-visible simplex by adding the point x as the peak
  /// vertex, also create new simplices
  void FillVisibleHull(PointRef x, const Deref& deref);
};

}  // namespace clarkson93

#endif  // CLARKSON93_TRIANGULATION_H_
