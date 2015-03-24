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
#ifndef EDELSBRUNNER96_LINE_WALKER_HPP_
#define EDELSBRUNNER96_LINE_WALKER_HPP_

#include <edelsbrunner96/line_walker.h>

#include <list>
#include <vector>
#include <edelsbrunner96/simplex.hpp>

namespace edelsbrunner96 {

/// Starting at centroid of simplex @p s_0, walk the triangulation in the
/// direction of @p p until the simplex containing @p p is found. Return
/// that simplex. Note that the returned simplex may be an "infinite" simplex,
/// i.e. a sentinal for a boundary facet of the convex hull.
template<typename Traits>
typename Traits::SimplexRef LineWalk(typename Traits::Storage& storage,
                                     typename Traits::SimplexRef s0_ref,
                                     const typename Traits::Point& pt) {
  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::PointRef PointRef;
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::Simplex Simplex;

  typedef Eigen::Matrix<Scalar, Traits::NDim, Traits::NDim> MatrixN;
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> VectorN;

  Simplex& s0 = storage[s0_ref];
  SimplexRef result = storage.NullSimplex();

  // median point of the starting simplex
  VectorN ps = VectorN::Zero();
  for (PointRef pr : s0.V) {
    ps += (1.0 / (Traits::NDim + 1)) * storage[pr];
  }

  // queue for the walk
  typedef std::pair<Scalar, SimplexRef> QueueElement;
  std::list<QueueElement> simplex_queue;

  // list of simplices we have already queued for walking (or already walked)
  std::list<SimplexRef> marked;

  // mark the initial simplex and queue it up
  s0.marked[simplex::LINE_WALK] = true;
  marked.push_back(s0_ref);
  simplex_queue.push_back(QueueElement(0, s0_ref));

  // while the simplex queue is not empty, expand simplices
  while (!simplex_queue.empty()) {
    QueueElement e = simplex_queue.front();
    Scalar s_min = e.first;
    SimplexRef simplex_ref = e.second;
    Simplex& simplex = storage[simplex_ref];
    simplex_queue.pop_front();

    // if this is an infinite simplex then just return it, as the walk has
    // left the triangulation (i.e. the query point is outside the current
    // hull)
    if (simplex.V[0] == storage.NullPoint()) {
      result = simplex_ref;
      break;
    }

    // for each facet, find the normal and offset relative to the starting
    // point, and compute the distance from the starting point to that plane
    // along the line segment we are traversing
    typedef std::pair<Scalar, int> FacetElement;
    typedef std::set<FacetElement> FacetSet;
    FacetSet facet_set;

    for (int i = 0; i < Traits::NDim + 1; i++) {
      MatrixN A;
      VectorN b;
      int k = 0;
      for (int j = 0; j < Traits::NDim + 1; j++) {
        if (j == i) {
          continue;
        }
        A.row(k++) = storage[simplex.V[j]] - ps;
      }
      b.fill(1);
      VectorN normal = A.fullPivLu().solve(b).normalized();
      Scalar offset = normal.dot(A.row(0));
      // orient the hyperplane so that it points "away" from the starting
      // point
      if (offset < 0) {
        normal = -normal;
        offset = -offset;
      }

      // compute the intersection of the facet with the line segment
      Scalar s = offset / (pt - ps).dot(normal);

      // if the intersection is not in the right direction, then skip it
      // Note(bialkowski): Since a vertices of a facet are ordered, and the
      // normal/offset of the vertex are always computed with respect to the
      // start point, s-value of the facet over which we crossed to enter this
      // simplex will always be computed the same as when it was evaluated
      // in the previous simplex. I.e. the A matrix is identical, the b-matrix
      // is identical, so there is no possibility of numerical stability issues.
      if (s <= s_min) {
        continue;
      }
      facet_set.insert(FacetElement(s, i));
    }

    // if no facets are found, then just continue
    if (facet_set.empty()) {
      continue;
    }

    FacetElement first_element = *facet_set.begin();
    if (first_element.first > 1) {
      result = simplex_ref;
      break;
    }

    // Starting with the first facet that we hit in this direction,
    // queue neighbors across all facets that are numerically the same distance
    // along this direction
    for (const FacetElement element : facet_set) {
      if (std::abs(element.first - first_element.first) < 1e-9) {
        SimplexRef neighbor_ref = simplex.N[element.second];
        Simplex& neighbor = storage[neighbor_ref];
        if (!neighbor.marked[simplex::LINE_WALK]) {
          simplex_queue.push_back(QueueElement(element.first, neighbor_ref));
          neighbor.marked[simplex::LINE_WALK] = true;
          marked.push_back(neighbor_ref);
        }
      }
    }
  }

  // clear markers
  for (SimplexRef s : marked) {
    storage[s].marked[simplex::LINE_WALK] = false;
  }
  return result;
}

template <typename Traits, typename InputIterator, typename OutputIterator>
void FeatureWalk(typename Traits::Storage& storage,
                 typename Traits::SimplexRef s_0,
                 InputIterator Vf_begin, InputIterator Vf_end,
                 OutputIterator out) {
  typedef typename Traits::PointRef PointRef;
  typedef typename Traits::SimplexRef SimplexRef;

  // number of vertices in the feature. The dimensonality of the feature is
  // n_vertices-1.
  int n_vertices = (Vf_end - Vf_begin);

  // queue of simplices to expand. We push new simplices to the back of the
  // queue, but we do not pop them from the front. Instead we maintain an
  // iterator into the queue up-to-which we have already processed. This way
  // we have a full list of marked simplices that we can unmark when we are
  // finished.
  std::list<SimplexRef> bfs_queue;
  storage[s_0].marked[simplex::FEATURE_WALK] = true;
  bfs_queue.push_back(s_0);

  // Vertices that are not part of the feature
  std::vector<PointRef> V;
  V.reserve(Traits::NDim + 1 - n_vertices);

  for (auto queue_iter = bfs_queue.begin(); queue_iter != bfs_queue.end();
       ++queue_iter) {
    SimplexRef s_ref = *queue_iter;

    // output this simplex
    *out++ = s_ref;

    // Store in V the sorted vertex references which are not part of the
    // common feature
    V.clear();
    std::set_difference(storage[s_ref].V.begin(),
                        storage[s_ref].V.end(),
                        Vf_begin, Vf_end, std::back_inserter(V));

    // verify that the feature is actually a feature of this simplex
    assert(V.size() + n_vertices == Traits::NDim+1);

    // for each vertex in that set, queue up the neighbor across that vertex,
    // unless it has already been queued.
    for (PointRef v : V) {
      SimplexRef neighbor_ref = storage[s_ref].NeighborAcross(v);
      if (!storage[neighbor_ref].marked[simplex::FEATURE_WALK]) {
        storage[neighbor_ref].marked[simplex::FEATURE_WALK] = true;
        bfs_queue.push_back(neighbor_ref);
      }
    }
  }

  // clear the mark
  for(SimplexRef s_ref : bfs_queue) {
    storage[s_ref].marked[simplex::FEATURE_WALK] = false;
  }
}

template <typename Traits, class OutputIterator>
void FuzzyWalk_(typename Traits::Storage& storage,
               const typename Traits::SimplexRef s_0,
               const typename Traits::Point& x_q,
               const typename Traits::Scalar epsilon,
               std::list<typename Traits::SimplexRef>& search_queue,
               OutputIterator out) {
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::Point Point;
  typedef typename Traits::PointRef PointRef;

  storage[s_0].marked[simplex::FUZZY_WALK] = true;
  search_queue.push_back(s_0);

  std::vector<PointRef> V_feature;
  V_feature.reserve(Traits::NDim+1);
  Point x_proj;
  for (auto queue_iter = search_queue.begin(); queue_iter != search_queue.end();
      ++queue_iter) {
    V_feature.clear();
    Scalar dist = SimplexDistance<Traits>(storage, x_q, *queue_iter, &x_proj,
                                          std::back_inserter(V_feature));

    // If the walk reaches a visible hull simplex, then we switch to a hull
    // walk to get all visible hull simplices
    if (storage[*queue_iter].V[0] == storage.NullPoint()) {
      // by convention, for an infinite simplex the SimplexDistance is the
      // distance to the hull facet, so we include it in the output set if
      // it is either visible or within epsilon distance of the query
      if (dist < epsilon || IsVisible<Traits>(storage, *queue_iter, x_q)) {
        *out++ = *queue_iter;
        // Switch to breadth-first search for visible hull simplices
        for (SimplexRef s_ref : storage[*queue_iter].N) {
          if (!storage[s_ref].marked[simplex::FUZZY_WALK]) {
            storage[s_ref].marked[simplex::FUZZY_WALK] = true;
            search_queue.push_back(s_ref);
          }
        }
      }
    } else {
      // If the simplex is within the fuzz distance, switch to breadth-first
      // search
      if (dist < epsilon) {
        *out++ = *queue_iter;
        for (SimplexRef s_ref : storage[*queue_iter].N) {
          if (!storage[s_ref].marked[simplex::FUZZY_WALK]) {
            storage[s_ref].marked[simplex::FUZZY_WALK] = true;
            search_queue.push_back(s_ref);
          }
        }
      // Otherise walk the nearest feature (depth-first)
      } else {
        std::list<SimplexRef> link;
        std::sort(V_feature.begin(), V_feature.end());
        FeatureWalk<Traits>(storage, *queue_iter, V_feature.begin(),
                            V_feature.end(), std::back_inserter(link));
        for (SimplexRef s_ref : link) {
          if (!storage[s_ref].marked[simplex::FUZZY_WALK]) {
            storage[s_ref].marked[simplex::FUZZY_WALK] = true;
            search_queue.push_back(s_ref);
          }
        }
      }
    }
  }

  for(SimplexRef s_ref : search_queue) {
    storage[s_ref].marked[simplex::FUZZY_WALK] = false;
  }
}

template <typename Traits, class OutputIterator>
void FuzzyWalk(typename Traits::Storage& storage,
               const typename Traits::SimplexRef s_0,
               const typename Traits::Point& x_q,
               const typename Traits::Scalar epsilon,
               OutputIterator out) {
  typedef typename Traits::SimplexRef SimplexRef;
  std::list<SimplexRef> search_queue;
  FuzzyWalk_<Traits>(storage, s_0, x_q, epsilon, search_queue, out);
}

template <class Traits>
bool IsVisible(typename Traits::Storage& storage,
               typename Traits::SimplexRef s_ref,
               const typename Traits::Point& x_q,
               typename Traits::Scalar epsilon) {
  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::PointRef PointRef;
  typedef typename Traits::Point Point;
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::Simplex Simplex;

  Simplex& simplex = storage[s_ref];
  SimplexRef interior_ref = simplex.N[0];
  Simplex& interior_simplex = storage[interior_ref];

  // find the index in the interior simplex vertex set of the vertex
  // which is not part of the common facet
  PointRef interior_peak_ref;
  std::set_difference(interior_simplex.V.begin(), interior_simplex.V.end(),
                      simplex.V.begin(), simplex.V.end(), &interior_peak_ref);

  // compute the normal and offset of the facet with respect to this
  // designated interior vertex
  typedef Eigen::Matrix<Scalar, Traits::NDim, Traits::NDim> MatrixN;
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> VectorN;
  const Point& interior_point = storage[interior_peak_ref];
  MatrixN A;
  for (int i = 1; i < Traits::NDim + 1; i++) {
    A.row(i - 1) = storage[simplex.V[i]] - interior_point;
  }
  VectorN b;
  b.fill(1.0);
  VectorN normal = A.fullPivLu().solve(b).normalized();
  Scalar offset = normal.dot(A.row(0));

  // orient the plane so that it "points away" from the interior or
  // the point set
  if (offset < 0) {
    normal = -normal;
    offset = -offset;
  }

  // now test the query point for visibility
  return (normal.dot(x_q - interior_point) - offset > -epsilon);
}

template<class Traits, class OutputIterator>
void GetVisibleHull(
    typename Traits::Storage& storage, typename Traits::SimplexRef s_0,
    const typename Traits::Point& x_q,
    OutputIterator out) {
  typedef typename Traits::Scalar Scalar;
  typedef typename Traits::PointRef PointRef;
  typedef typename Traits::Point Point;
  typedef typename Traits::SimplexRef SimplexRef;
  typedef typename Traits::Simplex Simplex;

  typedef Eigen::Matrix<Scalar, Traits::NDim, Traits::NDim> MatrixN;
  typedef Eigen::Matrix<Scalar, Traits::NDim, 1> VectorN;

  std::list<SimplexRef> visible_list;
  std::list<SimplexRef> marked_list;  // evaluated simplices

  // initialize the queue
  visible_list.push_back(s_0);
  storage[s_0].marked[simplex::VISIBLE_HULL_WALK] = true;
  marked_list.push_back(s_0);

  typename std::list<SimplexRef>::iterator iter = visible_list.begin();
  for (; iter != visible_list.end(); ++iter) {
    SimplexRef simplex_ref = *iter;
    Simplex& simplex = storage[simplex_ref];

    // for each un-evaluated neighbor, evaluate visibility, and if it is
    // visible add it to the result set and queue it for expansion.
    // Note(bialkowski): Because the designated NullPoint is assumed to
    // compare less than all other point references, all neighbors are hull
    // simplices except for the 0th neighbor.
    for (int i = 1; i < Traits::NDim + 1; i++) {
      SimplexRef neighbor_ref = simplex.N[i];
      Simplex& neighbor_simplex = storage[neighbor_ref];
      if (!neighbor_simplex.marked[simplex::VISIBLE_HULL_WALK]) {
        neighbor_simplex.marked[simplex::VISIBLE_HULL_WALK] = true;
        marked_list.push_back(neighbor_ref);
        SimplexRef interior_ref = neighbor_simplex.N[0];
        Simplex& interior_simplex = storage[interior_ref];

        // find the index in the interior simplex vertex set of the vertex
        // which is not part of the common facet
        PointRef interior_peak_ref;
        std::set_difference(
            interior_simplex.V.begin(), interior_simplex.V.end(),
            neighbor_simplex.V.begin(), neighbor_simplex.V.end(),
            &interior_peak_ref);

        // compute the normal and offset of the facet with respect to this
        // designated interior vertex
        typedef Eigen::Matrix<Scalar, Traits::NDim, Traits::NDim> MatrixN;
        typedef Eigen::Matrix<Scalar, Traits::NDim, 1> VectorN;
        const Point& interior_point = storage[interior_peak_ref];
        MatrixN A;
        for (int i = 1; i < Traits::NDim + 1; i++) {
          A.row(i - 1) = storage[neighbor_simplex.V[i]] - interior_point;
        }
        VectorN b;
        b.fill(1.0);
        VectorN normal = A.fullPivLu().solve(b).normalized();
        Scalar offset = normal.dot(A.row(0));

        // orient the plane so that it "points away" from the interior or
        // the point set
        if (offset < 0) {
          normal = -normal;
          offset = -offset;
        }

        // now test the query point for visibility
        if (normal.dot(x_q - interior_point) > offset) {
          visible_list.push_back(neighbor_ref);
        }
      }
    }
  }

  // clear all the marks
  for (SimplexRef s_ref : marked_list) {
    storage[s_ref].marked[simplex::VISIBLE_HULL_WALK] = false;
  }

  // output results
  for (SimplexRef s_ref : visible_list) {
    *out++ = s_ref;
  }
}

template <class Traits, class OutputIterator>
void BreadthFirstSearch(typename Traits::Storage& storage,
                        typename Traits::SimplexRef simplex_ref,
                        OutputIterator out) {
  typedef typename Traits::Simplex Simplex;
  typedef typename Traits::SimplexRef SimplexRef;

  std::list<SimplexRef> bfs_queue;
  bfs_queue.push_back(simplex_ref);
  storage[simplex_ref].marked[simplex::BFS_QUEUED] = true;

  typename std::list<SimplexRef>::iterator iter = bfs_queue.begin();
  for (; iter != bfs_queue.end(); ++iter) {
    SimplexRef simplex_ref = *iter;
    *out++ = simplex_ref;
    Simplex& simplex = storage[simplex_ref];
    for (SimplexRef neighbor_ref : simplex.N) {
      Simplex& neighbor = storage[neighbor_ref];
      if (!neighbor.marked[simplex::BFS_QUEUED]) {
        neighbor.marked[simplex::BFS_QUEUED] = true;
        bfs_queue.push_back(neighbor_ref);
      }
    }
  }

  for (SimplexRef simplex_ref : bfs_queue) {
    storage[simplex_ref].marked[simplex::BFS_QUEUED] = false;
  }
}

}  // namespace edelsbrunner96

#endif  // EDELSBRUNNER96_LINE_WALKER_HPP_
