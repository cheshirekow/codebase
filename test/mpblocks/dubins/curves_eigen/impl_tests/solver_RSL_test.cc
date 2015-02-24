#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mpblocks/dubins/curves_eigen/impl/SolutionRSL.hpp>


typedef Eigen::Matrix<double,3,1> State;

using namespace mpblocks;
typedef dubins::Path<double> Path;
typedef dubins::curves_eigen::Solver<dubins::RSL,double> Solver;

/**
 *
        *o-x        *  *
     *        x  *        *
    *          x*          *
    *          *x          *
     *        *  x        *
        *  *        xo-*
 */
TEST(SolverRSLTest, QuerySideBySide) {
  State q0{0, 2, 0};
  State q1{2, 0, 0};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::RSL, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_EQ(M_PI / 2, soln.s[0]);
  EXPECT_EQ(0, soln.s[1]);
  EXPECT_EQ(M_PI / 2, soln.s[2]);
}

/**
 *
        *o-xxxx                 *  *
     *        * xxx          *        *
    *          *   xxx      *          *
    *          *      xxx   *          *
     *        *          xxx *        *
        *  *                 xxxxo-*
 */
TEST(SolverRSLTest, QueryColinear) {
  State q0{0, 2, 0};
  State q1{4, 0, 0};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::RSL, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_LT(0, soln.s[0]);
  EXPECT_GT(M_PI / 2, soln.s[0]);
  EXPECT_LT(2, soln.s[1]);
  EXPECT_GT(4, soln.s[1]);
  EXPECT_LT(0, soln.s[2]);
  EXPECT_GT(M_PI / 2, soln.s[2]);
}

/**
 *
        *  *
     *        *
    *          *
    *          *
     *        *
        *o-*
     *        *
    *          *
    *          *
     *        *
        *  *
 */
TEST(SolverRSLTest, QueryDegenerate) {
  State q0{0, 0, 0};
  State q1{0, 0, 0};
  Path soln = Solver::solve(q0, q1, 1);

  EXPECT_EQ(dubins::RSL, soln.id);
  EXPECT_TRUE(soln.f);
  EXPECT_EQ(0, soln.s[0]);
  EXPECT_EQ(0, soln.s[1]);
  EXPECT_EQ(0, soln.s[2]);
}


