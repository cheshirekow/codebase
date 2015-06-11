#include <gtest/gtest.h>
#include "clarkson93/triangulation_impl.h"

static const int kNullPoint = -2;
static const int kAntiOrigin = -1;

typedef Eigen::Vector3d Point;

struct PointDeref {
 public:
  PointDeref(std::vector<Point>* point_store) : point_store_(point_store) {}

  Point& operator()(int i) const {
    return (*point_store_)[i];
  }

 private:
  std::vector<Point>* point_store_;
};

struct TestTraits {
  static const int kDim = 3;
  typedef double Scalar;
  typedef int PointRef;
  typedef PointDeref Deref;

  struct SimplexAllocator {
    clarkson93::Simplex<TestTraits>* Create() {
      return new clarkson93::Simplex<TestTraits>(kNullPoint);
    }

    void Free(clarkson93::Simplex<TestTraits>* s_ptr) {
      delete s_ptr;
    }
  };
};

typedef clarkson93::Triangulation<TestTraits> Triangulation;

TEST(TriangulationTest, BuildInitialTest) {
  TestTraits::SimplexAllocator simplex_allocator;
  Triangulation triangulation(kAntiOrigin, &simplex_allocator);

  std::vector<Point> point_store;
  PointDeref store_deref(&point_store);

  // canonical simplex
  point_store.push_back({0, 0, 0});
  for (int i = 0; i < 3; i++) {
    Point point = Point::Zero();
    point[i] = 1;
    point_store.emplace_back(point);
  }

  triangulation.BuildFromIL({0, 1, 2, 3}, store_deref);
}
