#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mpblocks/dubins/curves_eigen/impl/SolutionLRLb.hpp>


typedef Eigen::Matrix<double,3,1> State;

using namespace mpblocks;
typedef dubins::Path<double> Path;
typedef dubins::curves_eigen::Solver<dubins::LRLb,double> Solver;

/**
 *
        *  *        x  x        *  *
     *        *  x        x  *        *
    *          |x          x*          *
    *          o*          *o          *
     *        *  *        * | *        *
        *  *        *  *        *  *
 */
TEST(SolverLRLbTest, QueryCoincidentColinear) {
  State q0{0, 0, M_PI / 2};
  State q1{2, 0, -M_PI / 2};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::LRLb, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_EQ(0, soln.s[0]);
  EXPECT_EQ(M_PI, soln.s[1]);
  EXPECT_EQ(0, soln.s[2]);
}

/**
 *
        *  *        x  x        *  *
     *        *  x        x  *        *
    *          xx          xx          *
    *          xx          xx          *
     *        x  *        *   x        *
        *o-x        *  *        xxo-*
 */
TEST(SolverLRLbTest, QueryColinear) {
  State q0{0, 0, 0};
  State q1{4, 0, 0};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::LRLb, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_EQ(M_PI / 2, soln.s[0]);
  EXPECT_EQ(M_PI, soln.s[1]);
  EXPECT_EQ(M_PI / 2, soln.s[2]);
}

/**
 *
        *  *
     *        *
    *          *
    *          *
     *        *
        *o-*
 */
TEST(SolverLRLbTest, QueryDegenerate) {
  State q0{0, 0, 0};
  State q1{0, 0, 0};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::LRLb, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_EQ(0, soln.s[0]);
  EXPECT_EQ(0, soln.s[1]);
  EXPECT_EQ(0, soln.s[2]);
}

/**
 *
                  x  x
               x        x
              x          x
        *  *  x          x  *  *
     *        xx        xx        *
    *          x  *  *  x          *
    *          x        x          *
     *        x          x        *
        *o-x                xxo-*
 */
TEST(SolverLRLbTest, QueryCanonical) {
  State q0{0, 0, 0};
  State q1{3, 0, 0};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::LRLb, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_LT(M_PI / 2, soln.s[0]);
  EXPECT_GT(M_PI, soln.s[0]);
  EXPECT_LT(M_PI, soln.s[1]);
  EXPECT_GT(3*M_PI/2, soln.s[1]);
  EXPECT_LT(M_PI / 2, soln.s[2]);
  EXPECT_GT(M_PI, soln.s[2]);
}
