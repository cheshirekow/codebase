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

#ifndef EDELSBRUNNER96_TRIANGULATION_H_
#define EDELSBRUNNER96_TRIANGULATION_H_

#include <array>
#include <edelsbrunner96/induced_subcomplex.h>
#include <edelsbrunner96/simplex.h>

namespace edelsbrunner96 {

/// Initialize a triangulation with NDim+1 points and return the simplex
/**
 *  The triangulation is actually composed of NDim+2 simplices. In 2d this
 *  is drawn in ascii art below.
 *  @code
 *
 *                    (inf)             [.] simplex
 *                     |                (.) vertex
 *                     |
 *                     |(2)
 *                    / \
 *             [1]   /   \    [0]
 *                  /     \
 *                 /  [3]  \
 *(inf) ----------/_________\------------ (inf)
 *             (0)           (1)
 *
 *                    [2]
@endcode
 *  Note that there are (NDim=2 + 2) = 4 simplices (triangles) generated. The
 *  triangle formed by the three vertices is [3] = conv({(0),(1),(2)}). There
 *  are three additional meta-simplices. Each one is formed by two of the
 *  vertices, with the addition of the designated "infinite" vertex as the
 *  third. In this way the entire space of R^2 is covered. Note that these
 *  infinite simplices are degenerate: their circumcenter and circumradius
 *  are undefined. This is useful, however, to ensure that the entire space of
 *  R^2 is covered by an element in the tesselation.
 */
template<class Traits>
typename Traits::SimplexRef Triangulate(
    typename Traits::Storage& storage,
    std::initializer_list<typename Traits::PointRef> refs);

/// While the link of the specified point is not locally Delaunay, continue
/// flipping locally non-Delaunay facets.
template <class Traits>
typename Traits::SimplexRef Maintain(typename Traits::Storage& storage,
                                     typename Traits::PointRef point_ref,
                                     typename Traits::SimplexRef return_ref,
                                     std::list<Facet<Traits>>& link_facets);


/// Inserts a point outside of the hull of the current point set. Note that
/// @p simplex_ref must point to a hull facet which is visible by the point
/// to insert.
template<class Traits>
typename Traits::SimplexRef InsertOutside(typename Traits::Storage& storage,
                                   typename Traits::SimplexRef simplex_ref,
                                   typename Traits::PointRef point_ref);

/// Inserts a point into a simplex, performs 1-to-n+1 flip, and
/// performs the delaunay maintenance on the modified graph
template <class Traits>
typename Traits::SimplexRef InsertInside(
    typename Traits::Storage& storage, typename Traits::SimplexRef simplex_ref,
    typename Traits::PointRef point_ref);

/// Inserts a point into the triangulation, replacing the given simplex set
/// which intersect the new point
template <class Traits, class Iterator>
typename Traits::SimplexRef InsertReplace(
    typename Traits::Storage& storage,
    typename Traits::PointRef point_ref,
    Iterator S_begin, Iterator S_end);

/// Perform a fuzzy walk to get the set of simplices intersecting the query
/// point, then insert the point into the triangulation
template <typename Traits>
typename Traits::SimplexRef FuzzyWalkInsert(
    typename Traits::Storage& storage, const typename Traits::SimplexRef s_0,
    const typename Traits::PointRef x_ref,
    const typename Traits::Scalar epsilon);

}  // namespace edelsbrunner

#endif // EDELSBRUNNER96_TRIANGULATION_H_
