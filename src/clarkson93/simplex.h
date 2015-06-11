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
#ifndef CLARKSON93_SIMPLEX_H_
#define CLARKSON93_SIMPLEX_H_

#include <mpblocks/clarkson93.h>

namespace mpblocks {
namespace clarkson93 {

namespace simplex {

enum Bits {
  XVISIBLE_WALK,       ///< has been queued during the x-visible walk
  XV_HULL,             ///< is x-visible and hull
  HULL,                ///< used in hull enumeration
  HULL_QUEUED,         ///< used in hull enumeration
  HORIZON,             ///< is a member of the horizon set
  HORIZON_FILL,        ///< is a member of the set of simplices created to
                       ///  fill the empty horizon wedge
  S_WALK,              ///< simplices encountered in walk around a common
                       ///  (NDim-1) edge
  ENUMERATE_QUEUED,    ///< not used internally, for enumerating the hull
  ENUMERATE_EXPANDED,  ///< not used internally, for enumerating the hull
  SEARCH_QUEUED,       ///< not used internally, can be used for searches
  SEARCH_EXPANEDED,    ///< not used internally, can be used for searches
  NUM_BITS
};

}  // namespace simplex

typedef simplex::Bits SimplexBits;

/// encapsulates a vertex, simplex pair where the simplex is the neighbor
/// across from the specified vertex

/// A simplex is the convex hull of d+1 points in general position (IGP), i.e.
/// they do not lie on the same hyperplane
/**
 *  In order to implicitly maintain the convensions of Clarkson.compgeo93,
 *  we will order the vertices and facets of the simplex such that the i'th
 *  facet is opposite the i'th vertex (using the definition of clarkson), and
 *  thus is defined by the d vertices
 *  excluding the i'th. In addition, the i'th neighbor is the neighbor sharing
 *  the i'th facet and opposite the i'th vertex.
 *
 *  By the uniformity conventions of Clarkson.compgeo93, we'll reserve the
 *  0 index for the 'peak' vertex, and the 'base' facet
 *
 *  Because the 0'th vertex is the 'peak' vertex and the 'peak' vertex of
 *  a boundary simplex in the extended triangulation is the anti origin, the
 *  address stored in vertices[0] will be equal to the address of
 *  Triangulation::origin for the triangulation which owns this simplex.
 *
 *  The simplex also stores a normal vector \p n and an offset \p c defining
 *  a linear inequality whose hyperplane is coincident to the 'base' facet and
 *  oriented such that n' x < c is the half-space containing the 'peak' vertex
 */
template <class Traits>
struct SimplexBase : BitMember<SimplexBits, simplex::NUM_BITS> {
  // Typedefs
  // -----------------------------------------------------------------------
  static const unsigned int NDim = Traits::NDim;
  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::Point Point;
  typedef typename Traits::Simplex Simplex;
  typedef typename Traits::PointRef PointRef;
  typedef typename Traits::PointDeref PointDeref;
  typedef typename Traits::idx_t idx_t;

  typedef SimplexBase<Traits> SimplexBase_t;
  typedef SimplexBase<Traits> This_t;
  typedef HorizonRidge<Traits> HorizonRidge_t;

  enum Orientation { INSIDE, OUTSIDE };

  // Data Members
  // -----------------------------------------------------------------------
  PointRef V[NDim + 1];  ///< vertices of the simplex
  Simplex* N[NDim + 1];  ///< simplices which share a facet

  Point n;   ///< normal vector of base facet
  Scalar o;  ///< offset of base facet inequality hyperplane

  // Methods
  // -----------------------------------------------------------------------
  /// calculate the normal and offset of the constraint given the d vertices
  /// on it's base facet (does not compute orientation / sign)
  void calculateConstraint(PointDeref& deref);

  /// orient the constraint by ensuring that the point x satisfies it (i.e.
  /// @f$ n \cdot x \le c @f$ )
  void orientConstraint(const Point& x, Orientation orient = INSIDE);

  /// returns the distance of x to the base facet, used in walking the
  /// triangulation to x
  Scalar normalProjection(const Point& x);

  /// returns true if vertex[0] is the anti origin
  bool isInfinite(PointRef antiOrigin);

  /// returns true if x is on the inside of the base facet (i.e. x is in the
  /// same half space as the simplex)
  bool isVisible(const Point& x);
};

/// default simplex structure which contains no additional functionality
template <class Traits>
class DefaultSimplex : public SimplexBase<Traits> {};

}  // namespace clarkson93
}  // namespace mpblocks

#endif  // CLARKSON93_SIMPLEX_H_
