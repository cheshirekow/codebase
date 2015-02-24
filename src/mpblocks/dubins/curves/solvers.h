/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
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
 *  @date   Oct 24, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_EXAMPLES_DUBINS_CURVES_SOLVERS_H_
#define MPBLOCKS_EXAMPLES_DUBINS_CURVES_SOLVERS_H_

#include <Eigen/Dense>
#include <gtkmm.h>
#include <vector>

namespace mpblocks {
namespace examples {
namespace   dubins {

/// base class for solves
class Solver {
 protected:
  Eigen::Vector3d q[2];  ///< states
  Eigen::Vector2d c[3];  ///< centers
  double l[3];           ///< distances
  double r;              ///< radius

  Cairo::RefPtr<Cairo::Pattern> pat_q;
  Cairo::RefPtr<Cairo::Pattern> pat_R;
  Cairo::RefPtr<Cairo::Pattern> pat_L;
  Cairo::RefPtr<Cairo::Pattern> pat_S;

 public:
  Solver();
  virtual ~Solver() {}

  virtual double getDist(int i) = 0;

  /// solve for the shortest path and return the path length
  virtual double solve(Eigen::Vector3d q[2], double r) = 0;

  /// draw a diagram of the solution
  virtual void draw(const Cairo::RefPtr<Cairo::Context>& ctx) = 0;
};

/// solves shortest path with LRL primitives and "top" orientation of R between
/// the two L's
class SolverLRLa : public Solver {
 private:
  bool feasible;

 public:
  virtual ~SolverLRLa() {}
  virtual double getDist(int i);
  virtual double solve(Eigen::Vector3d q[2], double r);
  virtual void draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

/// solves shortest path with LRL primitives and "bottom" orientation of R
/// between the two L's
class SolverLRLb : public Solver {
 private:
  bool feasible;

 public:
  virtual ~SolverLRLb() {}
  virtual double getDist(int i);
  virtual double solve(Eigen::Vector3d q[2], double r);
  virtual void draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

/// solves shortest path with RLR primitives and "top" orientation of L between
/// the two R's
class SolverRLRa : public Solver {
 private:
  bool feasible;

 public:
  virtual ~SolverRLRa() {}
  virtual double getDist(int i);
  virtual double solve(Eigen::Vector3d q[2], double r);
  virtual void draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

/// solves shortest path with RLR primitives and "bottom" orientation of
/// L between the two R's
class SolverRLRb : public Solver {
 private:
  bool feasible;

 public:
  virtual ~SolverRLRb() {}
  virtual double getDist(int i);
  virtual double solve(Eigen::Vector3d q[2], double r);
  virtual void draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

/// solves shortest path with LSL primitives
class SolverLSL : public Solver {
 public:
  virtual ~SolverLSL() {}
  virtual double getDist(int i);
  virtual double solve(Eigen::Vector3d q[2], double r);
  virtual void draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

/// solves shortest path with LSR primitives
class SolverLSR : public Solver {
 private:
  bool feasible;

 public:
  virtual ~SolverLSR() {}
  virtual double getDist(int i);
  virtual double solve(Eigen::Vector3d q[2], double r);
  virtual void draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

/// solves shortest path with LSL primitives
class SolverRSL : public Solver {
 private:
  bool feasible;

 public:
  virtual ~SolverRSL() {}
  virtual double getDist(int i);
  virtual double solve(Eigen::Vector3d q[2], double r);
  virtual void draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

/// solves shortest path with LSR primitives
class SolverRSR : public Solver {
 public:
  virtual ~SolverRSR() {}
  virtual double getDist(int i);
  virtual double solve(Eigen::Vector3d q[2], double r);
  virtual void draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

}  // dubins
}  // examples
}  // mpblocks

#endif  // MPBLOCKS_EXAMPLES_DUBINS_CURVES_SOLVERS_H_
