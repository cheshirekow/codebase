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
 *  @date   Sept 22, 2014
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */
#ifndef EDELSBRUNNER96_LINE_WALKER_H_
#define EDELSBRUNNER96_LINE_WALKER_H_

#include <list>

namespace edelsbrunner96 {

/// Starting at the median point of simplex @p s_0, walk the triangulation in
/// the direction of @p p until the simplex containing @p p is found. Return
/// that simplex. Note that the returned simplex may be an "infinite" simplex,
/// i.e. a sentinal for a boundary facet of the convex hull.
template <typename Traits>
typename Traits::SimplexRef LineWalk(typename Traits::Storage& storage,
                                     typename Traits::SimplexRef s_0,
                                     const typename Traits::Point& p);

/// Given a sub-simplex feature and a simplex adjacent to that feature,
/// enumerate all simplices that are adjacent to that feature
/**
 *  A "feature" is a simplex s = conv(V), of |V| < NDim+1 vertices. It is a
 *  simplex of lower dimension than the embedding dimension of the
 *  triangulation. As an example, for a triangulation in 3 dimensions, a
 *  simplex is a tetrahedron, which is the convex hull of four vertices. The
 *  (3-1=2)-dimensional features of this simplex are it's triangle facets.
 *  The (3-2=1)-dimensional features of this simplex are it's line-segment
 *  edges. The (3-3=0)-dimensional features are it's vertices.
 *
 *  Given a simplex S, with vertex set V_S, and a feature with vertex set V_f,
 *  we identify the set of vertices V = V_S \ V_f. For each vertex v in V,
 *  the neighbor of S across from v shares all vertices V \ v. In particular,
 *  it shares V_f. We enumerate simplices of a common feature by breadth-first
 *  utilizing this fact.
 */
template <typename Traits, typename InputIterator, typename OutputIterator>
void FeatureWalk(typename Traits::Storage& storage,
                 typename Traits::SimplexRef s_0,
                 InputIterator Vf_begin, InputIterator Vf_end,
                 OutputIterator out);

/// Implementation, exposed so that the search_queue can be examined in tests
template <typename Traits, class OutputIterator>
void FuzzyWalk_(typename Traits::Storage& storage,
               const typename Traits::SimplexRef s_0,
               const typename Traits::Point& x_q,
               const typename Traits::Scalar epsilon,
               std::list<typename Traits::SimplexRef>& search_queue,
               OutputIterator out);

/// Given a simplex in a triangulation and a query point, walk the triangulation
/// in the direction of x_q until we find the set of simplices that the query
/// intersects
/**
 *  We return all simplices which are within @p epsilon distance from @p x_q.
 *  The search is a greedy walk through the triangulation. Starting at @p s_0,
 *  we find the feature of the simplex nearest to x_q. We then queue up all
 *  simplices that are adjacent to that feature. A feature is a sub-simplex
 *  such as a facet, edge, or lower-dimensional hull of some subset of the
 *  vertex set (down to a single vertex). For instance, if the query point is
 *  closest to a facet of s_0, we only queue up the neighbor of s_0 across
 *  that facet. If x_q is closest to a vertex of s_0, we queue up all simplices
 *  that share that vertex.
 *
 *  param storage   the storage model
 *  param s_0       the simplex to start the walk at
 *  param epsilon   the radius of fuzz for which we consider something to
 *                  intersect
 *  param x_q       the query point
 *  param[out] out  iterator where we write the output set of simplices
 */
template <typename Traits, class OutputIterator>
void FuzzyWalk(typename Traits::Storage& storage,
               const typename Traits::SimplexRef s_0,
               const typename Traits::Point& x_q,
               const typename Traits::Scalar epsilon,
               OutputIterator out);

/// Given a hull simplex, return true if it is visible by the query point
template<class Traits>
bool IsVisible(
    typename Traits::Storage& storage, typename Traits::SimplexRef s_ref,
    const typename Traits::Point& x_q,
    typename Traits::Scalar epsilon=0.0);

/// Starting at some x-visible hull simplex, return a list of all x-visible
/// hull simplices
template<class Traits, class  OutputIterator>
void GetVisibleHull(
    typename Traits::Storage& storage, typename Traits::SimplexRef s_0,
    const typename Traits::Point& x_q,
    OutputIterator out);

/// Get a list of all simplices by breadth first search starting at @p
/// simplex_ref
template<class Traits, class OutputIterator>
void BreadthFirstSearch(typename Traits::Storage& storage,
                        typename Traits::SimplexRef simplex_ref,
                        OutputIterator out);

}  // namespace edelsbrunner96

#endif  // EDELSBRUNNER96_LINE_WALKER_H_
