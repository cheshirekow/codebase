#include <iostream>
#include <gtest/gtest.h>

#include <edelsbrunner96.hpp>

namespace edelsbrunner96 {

template<typename Scalar_in, int NDim_in>
struct TestTraits {
  typedef TestTraits<Scalar_in, NDim_in> This;
  static const int NDim = NDim_in;

  typedef Scalar_in Scalar;
  typedef Eigen::Matrix<Scalar, NDim, 1> Point;
  typedef const Point* PointRef;

  typedef edelsbrunner96::SimplexBase<This> Simplex;
  typedef Simplex* SimplexRef;

  class Storage {
   public:
    const Point& operator[](const PointRef p) const {
      return *p;
    }

    PointRef NullPoint() {
      return nullptr;
    }

    Simplex& operator[](SimplexRef s) const {
      return *s;
    }

    SimplexRef NullSimplex() {
      return nullptr;
    }

   private:
    std::list<SimplexRef> free_;
    std::list<SimplexRef> alloced_;
  };
};

}  // namespace edelsbrunner96

TEST(SimplexTest, FillTest2d) {
  typedef edelsbrunner96::TestTraits<double, 2> Traits;
  typedef Traits::Simplex Simplex;
  typedef Traits::Point Point;

  // The canonical simplex in 2d
  Point vertices[] = {{0, 0}, {1, 0}, {0, 1}};
  std::vector<Point*> refs(3);
  for (int i = 0; i < 3; i++) {
    refs[i] = vertices + i;
  }
  Simplex simplex;
  simplex.SetVertices(vertices, vertices + 1, vertices + 2);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(simplex.V[i], vertices + i) << "For vertex" << i
                                          << " fill by list";
  }
  simplex.SetVertices(refs);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(simplex.V[i], vertices + i) << "For vertex" << i
                                          << " fill by vector";
  }
}

/// returns a vector of pointers from an array of points
template <typename Traits>
std::vector<typename Traits::PointRef> PointsToRefs(
    typename Traits::Point points[Traits::NDim + 1]) {
  std::vector<typename Traits::PointRef> refs(Traits::NDim + 1);
  for (int i = 0; i < Traits::NDim + 1; i++) {
    refs[i] = points + i;
  }
  return refs;
}

TEST(SimplexTest, CoordinatesOfTest2d) {
  typedef edelsbrunner96::TestTraits<double, 2> Traits;
  typedef Traits::Simplex Simplex;
  typedef Traits::Point Point;

  Simplex simplex;
  // The canonical simplex in 2d
  Point vertices[] = {{0, 0}, {1, 0}, {0, 1}};
  simplex.SetVertices(PointsToRefs<Traits>(vertices));

  // The canonical simplex is aligned to the euclidean axes
  // simplex-coordinates should be the same as euclidean coordinates
  std::vector<Point> test_points = {
      { 0, 0 },
      { 1, 0 },
      { 0, 1 },
      { 0.5, 0.5 },
      { 2, 0 },
      { 0, 2 },
      { 3, 3 } };

  Traits::Storage storage;
  for (uint32_t i = 0; i < test_points.size(); i++) {
    Point L_computed = simplex.CoordinatesOf(storage, test_points[i]);
    EXPECT_GT(1e-9, (L_computed - test_points[i]).squaredNorm())
        << " for test point: " << test_points[i].transpose();
  }
}

TEST(SimplexTest, CoordinatesOfTest3d) {
  typedef edelsbrunner96::TestTraits<double, 3> Traits;
  typedef Traits::Simplex Simplex;
  typedef Traits::Point Point;

  Simplex simplex;
  // The canonical simplex in 2d
  Point vertices[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  simplex.SetVertices(PointsToRefs<Traits>(vertices));

  // The canonical simplex is aligned to the euclidean axes
    // simplex-coordinates should be the same as euclidean coordinates
  std::vector<Point> test_points = {
      { 0, 0, 0 },
      { 1, 0, 0 },
      { 0, 1, 0 },
      { 0, 0, 1 },
      { 0.5, 0.5, 0.5 },
      { 2, 0, 0 },
      { 0, 2, 0 },
      { 0, 0, 2 },
      { 3, 3, 3 } };

  Traits::Storage storage;
  for (uint32_t i = 0; i < test_points.size(); i++) {
    Point L_computed = simplex.CoordinatesOf(storage, test_points[i]);
    EXPECT_GT(1e-9, (L_computed - test_points[i]).squaredNorm())
        << " for test point: " << test_points[i].transpose();
  }
}

TEST(SimplexTest, BarycentricCoordinatesTest2d) {
  typedef edelsbrunner96::TestTraits<double, 2> Traits;
  typedef Traits::Simplex Simplex;
  typedef Traits::Point Point;

  Simplex simplex;
  // The canonical simplex in 2d
  Point vertices[] = {{0, 0}, {1, 0}, {0, 1}};
  simplex.SetVertices(PointsToRefs<Traits>(vertices));

  // test points
  std::vector<Point> test_points = {
      { 0, 0 },
      { 1, 0 },
      { 0, 1 },
      { 0.5, 0.5 },
      { 2, 0 },
      { 0, 2 },
      { 3, 3 } };
  // expected barycentric coordinates
  std::vector<Eigen::Vector3d> expected = {
      { 1, 0, 0 },  // the first three points are the simplex corners and
      { 0, 1, 0 },  // so should have a non-zero element equal to unity
      { 0, 0, 1 },
      { 0, 0.5, 0.5 },
      { -1, 2, 0 },
      { -1, 0, 2 },
      { -5, 3, 3 }
  };

  Traits::Storage storage;
  for (uint32_t i = 0; i < test_points.size(); i++) {
    auto L_computed = simplex.BarycentricCoordinates(storage, test_points[i]);
    EXPECT_GT(1e-9, (L_computed - expected[i]).squaredNorm())
        << "   Barycentric : " << L_computed.transpose()
        << "  Expected : " << expected[i].transpose()
        << "\n   Cartesian : " << test_points[i].transpose();
  }
}

TEST(SimplexTest, BarycentricCoordinatesTest3d) {
  typedef edelsbrunner96::TestTraits<double, 3> Traits;
  typedef Traits::Simplex Simplex;
  typedef Traits::Point Point;

  Simplex simplex;
  // The canonical simplex in 2d
  Point vertices[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  simplex.SetVertices(PointsToRefs<Traits>(vertices));

  // test points
  std::vector<Point> test_points = {
      { 0, 0, 0 },
      { 1, 0, 0 },
      { 0, 1, 0 },
      { 0, 0, 1 },
      { 0.5, 0.5, 0.5 },
      { 2, 0, 0 },
      { 0, 2, 0 },
      { 0, 0, 2 },
      { 3, 3, 3 } };

  // expected barycentric coordinates
  std::vector<Eigen::Vector4d> expected = {
      { 1, 0, 0, 0 },  // the first four points are the simplex corners and
      { 0, 1, 0, 0 },  // so should have a non-zero element equal to unity
      { 0, 0, 1, 0 },
      { 0, 0, 0, 1 },
      { -0.5, 0.5, 0.5, 0.5 },
      { -1, 2, 0, 0 },
      { -1, 0, 2, 0 },
      { -1, 0, 0, 2 },
      { -8, 3, 3, 3 }
  };

  Traits::Storage storage;
  for (uint32_t i = 0; i < test_points.size(); i++) {
    auto L_computed = simplex.BarycentricCoordinates(storage, test_points[i]);
    EXPECT_GT(1e-9, (L_computed - expected[i]).squaredNorm())
        << "   Barycentric : " << L_computed.transpose()
        << "  Expected : " << expected[i].transpose()
        << "\n   Cartesian : " << test_points[i].transpose();
  }
}

TEST(SimplexTest, ContainsTest2d) {
  typedef edelsbrunner96::TestTraits<double, 2> Traits;
  typedef Traits::Simplex Simplex;
  typedef Traits::Point Point;

  Simplex simplex;
  // The canonical simplex in 2d
  Point vertices[] = {{0, 0}, {1, 0}, {0, 1}};
  simplex.SetVertices(PointsToRefs<Traits>(vertices));

  Traits::Storage storage;

  // test points
  std::vector<Point> test_points_inside = {
      { 0.2, 0.79 },
      { 0.5, 0.49 },
      { 0.8, 0.19 },
      { 0.3, 0.3 } };

  for (const Point& xq : test_points_inside) {
    EXPECT_TRUE(simplex.Contains(storage, xq)) << "  Point: " << xq.transpose();
  }

  std::vector<Point> test_points_outside = {
      { 0.2, 0.81 },
      { 0.5, 0.51 },
      { 0.8, 0.21 },
      { -1,  0.1 },
      { 0.1, -1 } };

  for (const Point& xq : test_points_outside) {
    EXPECT_FALSE(simplex.Contains(storage, xq))
        << "  Point: " << xq.transpose();
  }
}

TEST(SimplexTest, CircumcenterTest2d) {
  typedef edelsbrunner96::TestTraits<double, 2> Traits;
  typedef Traits::Simplex Simplex;
  typedef Traits::Point Point;

  Simplex simplex;

  std::vector<Point> test_centers = {{0, 0}, {1, 1}};
  for (const Point& center : test_centers) {
    // The canonical simplex in 2d
    Point vertices[3];
    for (int i = 0; i < 3; i++) {
      double theta = i * M_PI / 2.0;
      vertices[i] = center + Point(std::cos(theta), std::sin(theta));
    }

    simplex.SetVertices(PointsToRefs<Traits>(vertices));

    Traits::Storage storage;
    simplex.ComputeCenter(storage);

    EXPECT_GT(1e-9, (simplex.c - center).squaredNorm())
        << "For test center: " << center.transpose();
  }
}
