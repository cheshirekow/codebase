#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mpblocks/dubins/curves_eigen/impl/SolutionRSR.hpp>


typedef Eigen::Matrix<double,3,1> State;

using namespace mpblocks;
typedef dubins::Path<double> Path;
typedef dubins::curves_eigen::Solver<dubins::RSR,double> Solver;

/**
 *
        *o-xxxxxxxxxxxxxxxxxxxxxxo-*
     *        *              *        *
    *          *            *          *
    *          *            *          *
     *        *               *        *
        *  *                    *  *
 */
TEST(SolverRSRTest, QueryColinear) {
  State q0{0, 0, 0 / 2};
  State q1{2, 0, 0 / 2};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::RSR, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_EQ(0, soln.s[0]);
  EXPECT_EQ(2, soln.s[1]);
  EXPECT_EQ(0, soln.s[2]);
}

/**
 *
        *o-*
     *        *
    *          *
    *          *
     *        *
        *  *
 */
TEST(SolverRSRTest, QueryDegenerate) {
  State q0{0, 0, 0};
  State q1{0, 0, 0};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::RSR, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_EQ(0, soln.s[0]);
  EXPECT_EQ(0, soln.s[1]);
  EXPECT_EQ(0, soln.s[2]);
}

/**
 *

        x  x
     x        x
    |          o
    o          |
     *        *
        *  *
 */
TEST(SolverRSRTest, QuerySameCircle) {
  State q0{0, 0, M_PI / 2};
  State q1{2, 0, -M_PI / 2};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::RSR, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_EQ(M_PI / 2, soln.s[0]);
  EXPECT_EQ(0, soln.s[1]);
  EXPECT_EQ(M_PI / 2, soln.s[2]);
}

/**
 *
                      xxxxxxxx  x  x
        x  x  xxxxxxxx       *        x
     x        *             *          o
    |          *            *          |
    o          *             *        *
     *        *                 *  *
        *  *
 */
TEST(SolverRSRTest, QueryCanonical) {
  State q0{0, 0, M_PI/2};
  State q1{4, 1, -M_PI/2};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::RSR, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_LT(0, soln.s[0]);
  EXPECT_GT(M_PI / 2, soln.s[0]);
  EXPECT_LT(2, soln.s[1]);
  EXPECT_GT(3, soln.s[1]);
  EXPECT_LT(M_PI / 2, soln.s[2]);
  EXPECT_GT(M_PI, soln.s[2]);
}
