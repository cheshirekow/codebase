#include <gtest/gtest.h>
#include "clarkson93/triangulation_impl.h"

static const int kNullPoint = -2;
static const int kAntiOrigin = -1;

typedef Eigen::Vector3d Point;

struct PointDeref {
 public:
  Point& operator()(int i) {
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
}
