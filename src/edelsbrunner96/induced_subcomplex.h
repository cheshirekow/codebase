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
#ifndef EDELSBRUNNER96_INDUCED_SUBCOMPLEX_H_
#define EDELSBRUNNER96_INDUCED_SUBCOMPLEX_H_

#include <array>
#include <set>

namespace edelsbrunner96 {

/// The induced subcomplex of a facet between two simplices is the set of
/// all simplices in the triangulation whose vertices are composed of those
/// from the two simplices in question.
/**
 *  The concept is confusing at first, but this ascii art may help:
 *  @verbatim

     \
      \
       \
        \
         \ v3
          |\\         s4
          | \   \
          |  \     \
    s5    |   \       \
          |    \  s1     \
          |   v1\___________\ ________________________
          |     /           / v2
          | s2 /  s0     /
          |   /       /
          |  /     /
          | /   /
        v0|//       s3
          /
         /
        /
       /
      /
@endverbatim
 *
 *  In 2d a "simplex" is a triangle, and a "facet" is a line segment. Note the
 *  facet between simplices s0 and s1 (the line segment v1 -> v2). If we take
 *  all the vertices of the two simplices that
 *  border that facet (i.e. union of the vertices of s0 and s1), we get the
 *  set {v[0], v[1], v[2], v[3]}. The induced subcomplex of this facet is
 *  the set of all simplices whose three vertices are all found in that set.
 *  Namely, in this case, the induced subcomplex is the set {s0, s1, s2}.
 */
template <typename Traits>
class InducedSubcomplex {
 public:
  typedef typename Traits::Scalar     Scalar;
  typedef typename Traits::Storage    Storage;
  typedef typename Traits::PointRef   PointRef;
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::Simplex    Simplex;

  /// We do a breadthfirst search for simplices in the induced subcomplex.
  /// This data structure is what we queue in the search.
  struct BreadthFirstNode {
    SimplexRef s_ref;
    PointRef   v_missing;

    BreadthFirstNode(SimplexRef s_ref, PointRef v_missing)
        : s_ref(s_ref),
          v_missing(v_missing) {
    }
  };

  /// the (sorted) vertices of the subcomplex
  std::array<PointRef, Traits::NDim + 2> V;

  /// if S[i] is not null then it is the simplex in the triangulation composed
  /// of all the vertices in V except for the i'th
  std::array<SimplexRef, Traits::NDim + 2> S;

  /// the vertices of facet for which this subcomplex is induced.
  std::array<PointRef, Traits::NDim> f;

  /// the two vertices in V which are not part of the facet. e[i] is the
  /// vertex missing from s[i]
  std::array<PointRef, 2> e;

  /// the two simplices bounding the facet
  std::array<SimplexRef, 2>  s;

  /// Initialize the data structure with the two simplices and build the
  /// vertex set for the common facet
  void Init(Storage& storage, SimplexRef s0_ref, SimplexRef s1_ref);

  /// Initialize the data structure with the two simplices and build the
  /// vertex set for the common facet
  void Init(Storage& storage, const std::array<SimplexRef, 2>& s);

  /// Return true if the facet is locally regular
  bool IsLocallyRegular(Storage& storage);

  /// Fill the induced simplex by breadth-first search of the neighborhood
  /// about s[0] and s[1]
  void Build(Storage& storage);

  /// Return true if the facet is flippable
  /**
   *  A facet is flippable if all reflex edges are filled by a simplex in
   *  the induced subcomplex.
   */
  bool IsFlippable(Storage& storage);

  /// Given a simplex s which borders the hull of V
  /// (i.e. s shares a facet with hull(V)), then find the vertex of s which is
  /// not in V, and return it's index in S.V;
  uint8_t FindIndexOfPeakVertex(Simplex& s);

  /// Flip the subcomplex
  void Flip(Storage& storage);
};

}  // namespace edelsbrunner96

#endif  // EDELSBRUNNER96_INDUCED_SUBCOMPLEX_H_
