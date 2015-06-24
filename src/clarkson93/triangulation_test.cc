#include <gtest/gtest.h>
#include <cppformat/format.h>
#include "clarkson93/triangulation_impl.h"

using namespace clarkson93;

typedef Eigen::Vector3d Point;

static Point* const kAntiOrigin = static_cast<Point*>(nullptr) - 1;

struct TestTraits {
  static const int kDim = 3;
  typedef double Scalar;
  typedef int PointRef;

  /// we hang on to constructed simplices so that we can do some tests
  struct SimplexAllocator {
    SimplexAllocator() : freed_set(simplex::FREED) {}

    Simplex<TestTraits>* Create() {
      Simplex<TestTraits>* s_ptr = new Simplex<TestTraits>;
      allocated.push_back(s_ptr);
      return s_ptr;
    }

    void Free(Simplex<TestTraits>* s_ptr) { freed_set.Add(s_ptr); }

    void Clear() {
      for (Simplex<TestTraits>* s_ptr : allocated) {
        delete s_ptr;
      }
    }

    std::list<Simplex<TestTraits>*> allocated;
    BitMemberSet<simplex::Sets> freed_set;
  };
};

typedef Triangulation<TestTraits> TestTriangulation;

std::string VertexString(
    const std::array<Point*, TestTraits::kDim + 1>& vertices) {
  std::stringstream strm;
  auto iter = vertices.begin();
  strm << *iter;
  for (++iter; iter != vertices.end(); ++iter) {
    strm << ", " << *iter;
  }
  return strm.str();
}

std::string NeighborString(
    const std::array<Simplex<TestTraits>*, TestTraits::kDim + 1>& neighbors) {
  std::stringstream strm;
  auto iter = neighbors.begin();
  strm << fmt::format("{}", static_cast<void*>(*iter));
  for (++iter; iter != neighbors.end(); ++iter) {
    strm << ", " << fmt::format("{}", static_cast<void*>(*iter));
  }
  return strm.str();
}

std::string SimplexString(Simplex<TestTraits>* s_ptr) {
  return fmt::format(
      "Simplex {}\n"
      "  Vertices: {}\n"
      "  Neighbors: {}\n",
      static_cast<void*>(s_ptr), VertexString(s_ptr->V),
      NeighborString(s_ptr->N));
}

testing::AssertionResult EveryNeighborPointsBackToSelf(
    Simplex<TestTraits>* s_ptr) {
  for (Simplex<TestTraits>* neighbor_ptr : s_ptr->N) {
    bool found_self_in_neighbor = false;
    for (Simplex<TestTraits>* neighbor_ptr2 : neighbor_ptr->N) {
      if (neighbor_ptr2 == s_ptr) {
        found_self_in_neighbor = true;
        break;
      }
    }
    if (!found_self_in_neighbor) {
      return testing::AssertionFailure()
             << fmt::format("Neighbor {} doesn't point back to self {}\n",
                            static_cast<void*>(s_ptr),
                            static_cast<void*>(neighbor_ptr))
             << SimplexString(s_ptr) << SimplexString(neighbor_ptr);
    }
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult SelfReferentialPointersAreNonMemberVertices(
    Simplex<TestTraits>* s_ptr) {
  for (Simplex<TestTraits>* neighbor_ptr : s_ptr->N) {
    for (int i = 0; i < TestTraits::kDim + 1; i++) {
      if (neighbor_ptr->N[i] == s_ptr) {
        Point* vertex_id_in_neighbor = neighbor_ptr->V[i];
        const int index_in_self = s_ptr->GetIndexOf(vertex_id_in_neighbor);
        if (index_in_self > 0 && index_in_self < TestTraits::kDim + 1 &&
            s_ptr->V[index_in_self] == vertex_id_in_neighbor) {
          return testing::AssertionFailure()
                 << fmt::format(
                        "Neighbor {} of {} refers back to in across "
                        "from a vertex that it, itself, contains\n",
                        static_cast<void*>(s_ptr),
                        static_cast<void*>(neighbor_ptr))
                 << SimplexString(s_ptr) << SimplexString(neighbor_ptr);
        }
      }
    }
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult NeighborsHaveOneVertexInCommon(
    Simplex<TestTraits>* s_ptr) {
  for (Simplex<TestTraits>* neighbor_ptr : s_ptr->N) {
    std::vector<Point*> vset_intersection;
    VsetIntersection(*s_ptr, *neighbor_ptr,
                     std::back_inserter(vset_intersection));
    if (vset_intersection.size() != TestTraits::kDim) {
      return testing::AssertionFailure()
             << fmt::format(
                    "Neighbor {} of {} and itself share the wrong "
                    "number of vertices \n",
                    static_cast<void*>(s_ptr), static_cast<void*>(neighbor_ptr))
             << SimplexString(s_ptr) << SimplexString(neighbor_ptr);
    }
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult HasNoNullNeighbor(Simplex<TestTraits>* s_ptr) {
  for (Simplex<TestTraits>* neighbor_ptr : s_ptr->N) {
    if (!neighbor_ptr) {
      return testing::AssertionFailure()
             << fmt::format("Simplex {} has null neighbors\n",
                            static_cast<void*>(s_ptr)) << SimplexString(s_ptr);
    }
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult InfiniteSimplexHasOneFiniteNeighbor(
    Simplex<TestTraits>* s_ptr) {
  if (!IsInfinite(*s_ptr, kAntiOrigin)) {
    return testing::AssertionSuccess();
  }

  int num_finite = 0;
  for (Simplex<TestTraits>* neighbor_ptr : s_ptr->N) {
    if (!IsInfinite(*neighbor_ptr, kAntiOrigin)) {
      num_finite++;
    }
  }
  if (num_finite != 1) {
    return testing::AssertionFailure()
           << fmt::format("Infinite simplex {} has {} finite neighbors\n",
                          static_cast<void*>(s_ptr), num_finite)
           << SimplexString(s_ptr);
  } else {
    return testing::AssertionSuccess();
  }
}

#define RaiseIfFailed(expression)                           \
  /* scope protect */ {                                     \
    testing::AssertionResult assertion_result = expression; \
    if (!assertion_result) {                                \
      return assertion_result;                              \
    }                                                       \
  }

/// walks the entire triangulation and returns true if it is consistent.
/**
 *  Consistency is defined as follows:
 *    * Each neighbor of a simplex points back to that simplex
 *    * The vertex associated with a simplex in it's neighbors mapping is
 *      not a member of that simplex
 *    * Each neighbor of a simplex has exactly one vertex not in common
 *    * There are no empty neighbor pointers
 *    * If a simplex is infinite, then it has exactly one finite neighbor
 */
testing::AssertionResult TriangulationIsConsistent(
    TestTraits::SimplexAllocator* allocator) {
  for (Simplex<TestTraits>* s_ptr : allocator->allocated) {
    // skip any that were freed
    if (allocator->freed_set.IsMember(*s_ptr)) {
      continue;
    }

    EXPECT_TRUE(HasNoNullNeighbor(s_ptr));
    EXPECT_TRUE(EveryNeighborPointsBackToSelf(s_ptr));
    EXPECT_TRUE(SelfReferentialPointersAreNonMemberVertices(s_ptr));
    EXPECT_TRUE(NeighborsHaveOneVertexInCommon(s_ptr));
    EXPECT_TRUE(InfiniteSimplexHasOneFiniteNeighbor(s_ptr));
  }
  return testing::AssertionSuccess();
}

TEST(TriangulationTest, InitialTriangulationOfCanoninicalSimplexIsConsistent) {
  TestTraits::SimplexAllocator simplex_allocator;
  TestTriangulation triangulation(kAntiOrigin, &simplex_allocator);

  std::vector<Point> point_store;

  // canonical simplex
  point_store.push_back({0, 0, 0});
  for (int i = 0; i < 3; i++) {
    Point point = Point::Zero();
    point[i] = 1;
    point_store.emplace_back(point);
  }

  triangulation.BuildFromIL({point_store.data() + 0, point_store.data() + 1,
                             point_store.data() + 2, point_store.data() + 3});

  // this first simplices in the list should be the origin, followed by the
  // origin's neighbors in order
  std::vector<Simplex<TestTraits>*> simplices(
      simplex_allocator.allocated.begin(), simplex_allocator.allocated.end());
  // we should have allocated NDim+1 simplices
  EXPECT_EQ(TestTraits::kDim + 2, simplices.size());

  EXPECT_EQ(
      (std::array<Point*, 4>({point_store.data() + 0, point_store.data() + 1,
                              point_store.data() + 2, point_store.data() + 3})),
      simplices[0]->V);
  for (int i = 0; i < TestTraits::kDim + 1; i++) {
    EXPECT_EQ(simplices[0]->N[i], simplices[i + 1]) << " for neighbor " << i;
  }

  // these should all be infinite
  for (int i = 0; i < TestTraits::kDim; i++) {
    EXPECT_EQ(kAntiOrigin, simplices[i + 1]->V[0]) << " for simplex" << i;
  }

  EXPECT_TRUE(TriangulationIsConsistent(&simplex_allocator));
}

TEST(TriangulationTest, TriangulationOfCircleIsConsistent) {
  TestTraits::SimplexAllocator simplex_allocator;
  TestTriangulation triangulation(kAntiOrigin, &simplex_allocator);

  std::vector<Point> point_store;

  // several points that all lie on a sphere
  for (double i : {-1, 0, 1}) {
    for (double j : {-1, 0, 1}) {
      for (double k : {-1, 0, 1}) {
        if (i != 0 || j != 0 || k != 0) {
          point_store.emplace_back(Point({i, j, k}).normalized());
        }
      }
    }
  }

  triangulation.BuildFromIL({point_store.data() + 0, point_store.data() + 1,
                             point_store.data() + 2, point_store.data() + 3});
  ASSERT_TRUE(TriangulationIsConsistent(&simplex_allocator));
  for (int i = 4; i < point_store.size(); i++) {
    triangulation.Insert(point_store.data() + i);
    ASSERT_TRUE(TriangulationIsConsistent(&simplex_allocator));
  }
}
