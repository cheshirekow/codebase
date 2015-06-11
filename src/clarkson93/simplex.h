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
#ifndef CLARKSON93_Simplex_H_
#define CLARKSON93_Simplex_H_

#include <array>
#include <cstdint>
#include <clarkson93/bit_member.h>
#include <Eigen/Dense>

namespace clarkson93 {

namespace simplex {

/// ID's for different sets that a particular simplex may be a member of
/// during various different phases of the algorithm
enum Sets {
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

enum Orientation { INSIDE, OUTSIDE };

}  // namespace simplex

typedef simplex::Sets SimplexSets;

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
 *  In this second version of the Simplex structure we do not maintain the
 *  zeroth vertex/Neighbor pair as the designated peak vertex, base facet.
 *  Instead we store the vertex references in sorted order so that we can
 *  efficiently perform queries regarding various dimensional facets of
 *  simplex.
 *
 *  The simplex also stores a normal vector \p n and an offset \p c defining
 *  a linear inequality whose hyperplane is coincident to the 'base' facet and
 *  oriented such that n' x < c is the half-space containing the 'peak' vertex
 */
template <class Traits>
struct Simplex : public BitMember<simplex::Sets, simplex::NUM_BITS> {
  // Typedefs
  // -----------------------------------------------------------------------
  static const int kDim = Traits::kDim;
  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::PointRef PointRef;
  typedef Eigen::Matrix<Scalar, kDim, 1> Point;

  // Data Members
  // -----------------------------------------------------------------------
  int8_t i_peak;  ///< index of the peak vertex
  // TODO(josh): compare performance between std::array and raw buffer
  std::array<PointRef, kDim + 1> V;  ///< vertices of the simplex
  std::array<Simplex*, kDim + 1> N;  ///< simplices which share a facet

  Point n;   ///< normal vector of base facet
  Scalar o;  ///< offset of base facet inequality hyperplane

  // Member Functions
  // -----------------------------------------------------------------------
  Simplex(PointRef null_point) : i_peak(0), o(0) {
    for (int i = 0; i < kDim + 1; i++) {
      V[i] = null_point;
      N[i] = nullptr;
    }
    n.fill(0);
  }

  Simplex<Traits>* GetNeighborAcross(PointRef vertex) {
    return N[GetIndexOf(vertex)];
  }

  void SetNeighborAcross(PointRef vertex, Simplex<Traits>* neighbor) {
    N[GetIndexOf(vertex)] = neighbor;
  }

  Simplex<Traits>* GetPeakNeighbor() {
    return N[i_peak];
  }

  PointRef GetPeakVertex() {
    return V[i_peak];
  }

  int8_t GetIndexOf(PointRef v) {
    return std::lower_bound(V.begin(), V.end(), v) - V.begin();
  }
};

// Readability functions
// ---------------------------------------------------------
template <class Traits>
const std::array<Simplex<Traits>*, Traits::kDim>& Neighborhood(
    const Simplex<Traits>& s) {
  return s.N;
}

template <class Traits>
const std::array<typename Traits::PointRef, Traits::kDim>& Vertices(
    const Simplex<Traits>& s) {
  return s.V;
}

/// simultaneously construct the set intersection and the set symmetric
/// difference of simplex_a and simplex_b
template <class Traits, typename Output1, typename Output2, typename Output3>
void VsetSplit(const Simplex<Traits>& simplex_a,
               const Simplex<Traits>& simplex_b, Output1 a_only, Output2 b_only,
               Output3 intersect);

/// builds a list of vertices common to both simplices
template <class Traits, typename Output>
void VsetIntersection(const Simplex<Traits>& simplex_a,
                      const Simplex<Traits>& simplex_b, Output intersect);

/// builds a list of neighbors that share a facet... this is the set of
/// neighbors who are accross from a vertex v where v is not in the
/// facet set
/**
 *  @param  simplex the simplex to query
 *  @param  feature   a range-iterable container of vertex references
 *  @param  outiter output iterator of neighbor simplices
 */
template <class Traits, class Container, class Output>
void GetNeighborsSharing(const Simplex<Traits>& simplex,
                         const Container& feature, Output out_iter);

// Mutators
// -----------------------------------------------------------------------
/// after filling vertices, lay them out in sorted order
template <class Traits>
void SortVertices(Simplex<Traits>* simplex);

/// compute the base facet normal and offset
template <class Traits, class Deref>
void ComputeBase(Simplex<Traits>* simplex, const Deref& deref);

/// orient the base facete normal by ensuring that the point x
/// lies on the appropriate half-space
/// @f$ n \cdot x \le c @f$ )
template <class Traits, class Point>
void OrientBase(Simplex<Traits>* simplex, const Point& x,
                simplex::Orientation orient = simplex::INSIDE);

// point queries
// -----------------------------------------------------------------------

/// returns the distance of x to the base facet, used in walking the
/// triangulation to x
template <class Traits, class Point>
typename Traits::Scalar NormalProjection(const Simplex<Traits>& simplex,
                                         const Point& x);

/// returns true if the base vertex is the anti origin
template <class Traits>
bool IsInfinite(const Simplex<Traits>& simplex,
                typename Traits::PointRef anti_origin);

/// returns true if x is on the inside of the base facet (i.e. x is in the
/// same half space as the simplex)
template <class Traits, class Point>
bool IsVisible(const Simplex<Traits>& simplex, const Point& x);

}  // namespace clarkson93

#endif  // CLARKSON93_Simplex_H_
