#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mpblocks/dubins/curves_eigen/impl/SolutionLSL.hpp>


typedef Eigen::Matrix<double,3,1> State;

using namespace mpblocks;
typedef dubins::Path<double> Path;
typedef dubins::curves_eigen::Solver<dubins::LSL,double> Solver;

/**
 *
        *  *                    *  *
     *        *              *        *
    *          *            *          *
    *          *            *          *
     *        *               *        *
        *o-xxxxxxxxxxxxxxxxxxxxxxo-*
 */
TEST(SolverLSLTest, QueryColinear) {
  State q0{0, 0, 0 / 2};
  State q1{2, 0, 0 / 2};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::LSL, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_EQ(0, soln.s[0]);
  EXPECT_EQ(2, soln.s[1]);
  EXPECT_EQ(0, soln.s[2]);
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
TEST(SolverLSLTest, QueryDegenerate) {
  State q0{0, 0, 0};
  State q1{0, 0, 0};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::LSL, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_EQ(0, soln.s[0]);
  EXPECT_EQ(0, soln.s[1]);
  EXPECT_EQ(0, soln.s[2]);
}

/**
 *

        *  *
     *        *
    o          |
    |          o
     x        x
        xxxx
 */
TEST(SolverLSLTest, QuerySameCircle) {
  State q0{0, 0, -M_PI / 2};
  State q1{2, 0, M_PI / 2};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::LSL, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_EQ(M_PI / 2, soln.s[0]);
  EXPECT_EQ(0, soln.s[1]);
  EXPECT_EQ(M_PI / 2, soln.s[2]);
}

/**
 *
                                *  *
        *  *                 *        *
     *        *             *          |
    o          *            *          o
    |          *             *        x
     x        *        xxxxxxxxxxxxx
        xxxxxxxxxxxxxxx
 */
TEST(SolverLSLTest, QueryCanonical) {
  State q0{0, 0, -M_PI/2};
  State q1{4, 1, M_PI/2};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::LSL, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_LT(M_PI / 2, soln.s[0]);
  EXPECT_GT(M_PI, soln.s[0]);
  EXPECT_LT(2, soln.s[1]);
  EXPECT_GT(3, soln.s[1]);
  EXPECT_LT(0, soln.s[2]);
  EXPECT_GT(M_PI / 2, soln.s[2]);
}
