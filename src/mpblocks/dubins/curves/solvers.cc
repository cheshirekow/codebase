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

#include <iostream>
#include "solvers.h"

namespace mpblocks {
namespace examples {
namespace   dubins {

Solver::Solver() {
  pat_q = Cairo::SolidPattern::create_rgb(0, 0, 0);
  pat_R = Cairo::SolidPattern::create_rgba(0, 0, 1, 0.5);
  pat_L = Cairo::SolidPattern::create_rgba(0, 0.8, 0, 0.5);
  pat_S = Cairo::SolidPattern::create_rgba(0.5, 188 / 255.0, 66 / 255.0, 0.5);
}

void drawArc(const Cairo::RefPtr<Cairo::Context>& ctx, const Eigen::Vector2d& c,
             double r, double a1, double a2) {
  Eigen::Vector2d v;
  v << r* cos(a1), r * sin(a1);
  v += c;

  ctx->move_to((double)v[0], (double)v[1]);
  ctx->arc((double)c[0], (double)c[1], r, a1, a2);
}

void drawCircle(const Cairo::RefPtr<Cairo::Context>& ctx,
                const Eigen::Vector2d& c, double r) {
  ctx->move_to((double)c[0] + r, (double)c[1]);
  ctx->arc((double)c[0], (double)c[1], r, 0, 2 * M_PI);
}

/// ensures that a is in [-pi, pi]
double clampRadian(double a) {
  while (a > M_PI) a -= 2 * M_PI;
  while (a < -M_PI) a += 2 * M_PI;

  return a;
}

/// returns the counter clockwise (left) distance from a to b
double counterClockwiseArc(double a, double b) {
  if (b > a)
    return b - a;
  else
    return 2 * M_PI + b - a;
}

/// returns the clockwise (right) distance from a to b
double clockwiseArc(double a, double b) {
  if (a > b)
    return a - b;
  else
    return 2 * M_PI + a - b;
}

double SolverLRLa::getDist(int i) { return 180 * l[i] / M_PI; }

double SolverLRLa::solve(Eigen::Vector3d qin[2], double rin) {
  using namespace std;
  using namespace Eigen;

  for (int i = 0; i < 2; i++) q[i] = qin[i];
  r = rin;

  Vector2d x, v, dc;
  double a, b, d;

  // calculate the center of the circle to which q1 is tangent
  x << q[0][0], q[0][1];
  v << -sin(q[0][2]), cos(q[0][2]);
  c[0] = x + r * v;

  // calculate the center of the circle to which q2 is tangent
  x << q[1][0], q[1][1];
  v << -sin(q[1][2]), cos(q[1][2]);
  c[1] = x + r * v;

  // the distance between the centers of these two circles
  d = (c[0] - c[1]).norm();

  // if the distance is too large, then this primitive is not the solution,
  // and we can bail here
  if (d > 4 * r) {
    for (int i = 0; i < 3; i++) l[i] = 0;
    feasible = false;
    return std::numeric_limits<double>::max();
  }

  feasible = true;

  if (d == 0) {
    Eigen::Map<Eigen::Vector3d>(l).fill(0);
    return 0;
  }

  // the base angle of the isosceles triangle whose vertices are the centers
  // of the the three circles, note acos returns [0,pi]
  a = acos(d / (4 * r));

  // create a clockwise rotation of magnitude alpha
  Rotation2Dd R(-a);

  // we find the third vertex of this triangle by taking the vector between
  // the two circle centers, normalizing it, and rotating it by alpha, and
  // scaling it to magnitude 2r, then it points from the center of
  // one the circle tangent to q1 to the third vertex
  c[2] = c[0] + R * (c[1] - c[0]).normalized() * 2 * r;

  // calculate the arc distance we travel on the first circle
  dc = c[2] - c[0];                       //< vector between centers of circles
  b = atan2(dc[1], dc[0]);                //< angle of that vector
  d = clampRadian(q[0][2] - M_PI / 2.0);  //< angle of vector from center to q1
  l[0] = counterClockwiseArc(d, b);       //< ccwise distance

  // calculate the arc distance we travel on the second circle
  dc = c[2] - c[1];                       //< vector between centers of circles
  b = atan2(dc[1], dc[0]);                //< angle of that vector
  d = clampRadian(q[1][2] - M_PI / 2.0);  //< angle of vector from center to q1
  l[1] = counterClockwiseArc(b, d);       //< ccwise distance

  // calculate the arc distance we travel on the third circle
  l[2] = M_PI - 2 * a;

  // sum the resulting segments
  double dist = 0;
  for (int i = 0; i < 3; i++) dist += l[i];

  return r * dist;
}

void SolverLRLa::draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  using namespace Eigen;
  ctx->set_source(pat_L);

  drawCircle(ctx, c[0], r);
  ctx->stroke();

  drawCircle(ctx, c[1], r);
  ctx->stroke();

  if (feasible) {
    ctx->set_source(pat_R);
    drawCircle(ctx, c[2], r);
    ctx->stroke();
  }

  ctx->unset_dash();
  ctx->set_source(pat_q);

  if (!feasible) return;

  double a0, a1, a2;

  a0 = clampRadian(q[0][2] - M_PI / 2.0);
  a1 = clampRadian(q[1][2] - M_PI / 2.0);
  a2 = clampRadian(a0 + l[0] + M_PI);

  ctx->move_to(q[0][0], q[0][1]);
  ctx->arc(c[0][0], c[0][1], r, a0, a0 + l[0]);
  ctx->arc_negative(c[2][0], c[2][1], r, a2, a2 - l[2]);
  ctx->arc(c[1][0], c[1][1], r, a1 - l[1], a1);

  ctx->stroke();
}

double SolverLRLb::getDist(int i) { return 180 * l[i] / M_PI; }

double SolverLRLb::solve(Eigen::Vector3d qin[2], double rin) {
  using namespace std;
  using namespace Eigen;

  for (int i = 0; i < 2; i++) q[i] = qin[i];
  r = rin;

  Vector2d x, v, dc;
  double a, b, d;

  // calculate the center of the circle to which q1 is tangent
  x << q[0][0], q[0][1];
  v << -sin(q[0][2]), cos(q[0][2]);
  c[0] = x + r * v;

  // calculate the center of the circle to which q2 is tangent
  x << q[1][0], q[1][1];
  v << -sin(q[1][2]), cos(q[1][2]);
  c[1] = x + r * v;

  // the distance between the centers of these two circles
  d = (c[0] - c[1]).norm();

  // if the distance is too large, then this primitive is not the solution,
  // and we can bail here
  if (d > 4 * r) {
    for (int i = 0; i < 3; i++) l[i] = 0;
    feasible = false;
    return std::numeric_limits<double>::max();
  }

  feasible = true;

  if (d == 0) {
    Eigen::Map<Eigen::Vector3d>(l).fill(0);
    return 0;
  }

  // the base angle of the isosceles triangle whose vertices are the centers
  // of the the three circles, note acos returns [0,pi]
  a = acos(d / (4 * r));

  // create a clockwise rotation of magnitude alpha
  Rotation2Dd R(-a);

  // we find the third vertex of this triangle by taking the vector between
  // the two circle centers, normalizing it, and rotating it by alpha, and
  // scaling it to magnitude 2r, then it points from the center of
  // one the circle tangent to q1 to the third vertex
  c[2] = c[1] + R * (c[0] - c[1]).normalized() * 2 * r;

  // calculate the arc distance we travel on the first circle
  dc = c[2] - c[0];                       //< vector between centers of circles
  b = atan2(dc[1], dc[0]);                //< angle of that vector
  d = clampRadian(q[0][2] - M_PI / 2.0);  //< angle of vector from center to q1
  l[0] = counterClockwiseArc(d, b);       //< ccwise distance

  // calculate the arc distance we travel on the second circle
  dc = c[2] - c[1];                       //< vector between centers of circles
  b = atan2(dc[1], dc[0]);                //< angle of that vector
  d = clampRadian(q[1][2] - M_PI / 2.0);  //< angle of vector from center to q1
  l[1] = counterClockwiseArc(b, d);       //< ccwise distance

  // calculate the arc distance we travel on the third circle
  l[2] = M_PI + 2 * a;

  // sum the resulting segments
  double dist = 0;
  for (int i = 0; i < 3; i++) dist += l[i];

  return r * dist;
}

void SolverLRLb::draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  ctx->set_source(pat_L);

  drawCircle(ctx, c[0], r);
  ctx->stroke();

  drawCircle(ctx, c[1], r);
  ctx->stroke();

  if (feasible) {
    ctx->set_source(pat_R);
    drawCircle(ctx, c[2], r);
    ctx->stroke();
  }

  ctx->unset_dash();
  ctx->set_source(pat_q);

  if (!feasible) return;

  double a0, a1, a2;

  a0 = clampRadian(q[0][2] - M_PI / 2.0);
  a1 = clampRadian(q[1][2] - M_PI / 2.0);
  a2 = clampRadian(a0 + l[0] + M_PI);

  ctx->move_to(q[0][0], q[0][1]);
  ctx->arc(c[0][0], c[0][1], r, a0, a0 + l[0]);
  ctx->arc_negative(c[2][0], c[2][1], r, a2, a2 - l[2]);
  ctx->arc(c[1][0], c[1][1], r, a1 - l[1], a1);

  ctx->stroke();
}

double SolverRLRa::getDist(int i) { return 180 * l[i] / M_PI; }

double SolverRLRa::solve(Eigen::Vector3d qin[2], double rin) {
  using namespace std;
  using namespace Eigen;

  for (int i = 0; i < 2; i++) q[i] = qin[i];
  r = rin;

  Vector2d x, v, dc;
  double a, b, d;

  // calculate the center of the circle to which q1 is tangent
  x << q[0][0], q[0][1];
  v << sin(q[0][2]), -cos(q[0][2]);
  c[0] = x + r * v;

  // calculate the center of the circle to which q2 is tangent
  x << q[1][0], q[1][1];
  v << sin(q[1][2]), -cos(q[1][2]);
  c[1] = x + r * v;

  // the distance between the centers of these two circles
  d = (c[0] - c[1]).norm();

  // if the distance is too large, then this primitive is not the solution,
  // and we can bail here
  if (d > 4 * r) {
    for (int i = 0; i < 3; i++) l[i] = 0;
    feasible = false;
    return std::numeric_limits<double>::max();
  }

  feasible = true;

  if (d == 0) {
    Eigen::Map<Eigen::Vector3d>(l).fill(0);
    return 0;
  }

  // the base angle of the isosceles triangle whose vertices are the centers
  // of the the three circles, note acos returns [0,pi]
  a = acos(d / (4 * r));

  // create a clockwise rotation of magnitude alpha
  Rotation2Dd R(-a);

  // we find the third vertex of this triangle by taking the vector between
  // the two circle centers, normalizing it, and rotating it by alpha, and
  // scaling it to magnitude 2r, then it points from the center of
  // one the circle tangent to q1 to the third vertex
  c[2] = c[0] + R * (c[1] - c[0]).normalized() * 2 * r;

  // calculate the arc distance we travel on the first circle
  dc = c[2] - c[0];                       //< vector between centers of circles
  b = atan2(dc[1], dc[0]);                //< angle of that vector
  d = clampRadian(q[0][2] + M_PI / 2.0);  //< angle of vector from center to q1
  l[0] = clockwiseArc(d, b);              //< ccwise distance

  // calculate the arc distance we travel on the second circle
  dc = c[2] - c[1];                       //< vector between centers of circles
  b = atan2(dc[1], dc[0]);                //< angle of that vector
  d = clampRadian(q[1][2] + M_PI / 2.0);  //< angle of vector from center to q1
  l[1] = clockwiseArc(b, d);              //< ccwise distance

  // calculate the arc distance we travel on the third circle
  l[2] = M_PI + 2 * a;

  // sum the resulting segments
  double dist = 0;
  for (int i = 0; i < 3; i++) dist += l[i];

  return r * dist;
}

void SolverRLRa::draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  ctx->set_source(pat_R);

  drawCircle(ctx, c[0], r);
  ctx->stroke();

  drawCircle(ctx, c[1], r);
  ctx->stroke();

  if (feasible) {
    ctx->set_source(pat_L);
    drawCircle(ctx, c[2], r);
    ctx->stroke();
  }

  ctx->unset_dash();
  ctx->set_source(pat_q);

  if (!feasible) return;

  double a0, a1, a2;

  a0 = clampRadian(q[0][2] + M_PI / 2.0);
  a1 = clampRadian(q[1][2] + M_PI / 2.0);
  a2 = clampRadian(a0 - l[0] + M_PI);

  ctx->move_to(q[0][0], q[0][1]);
  ctx->arc_negative(c[0][0], c[0][1], r, a0, a0 - l[0]);
  ctx->arc(c[2][0], c[2][1], r, a2, a2 + l[2]);
  ctx->arc_negative(c[1][0], c[1][1], r, a1 + l[1], a1);

  ctx->stroke();
}

double SolverRLRb::getDist(int i) { return 180 * l[i] / M_PI; }

double SolverRLRb::solve(Eigen::Vector3d qin[2], double rin) {
  using namespace std;
  using namespace Eigen;

  for (int i = 0; i < 2; i++) q[i] = qin[i];
  r = rin;

  Vector2d x, v, dc;
  double a, b, d;

  // calculate the center of the circle to which q1 is tangent
  x << q[0][0], q[0][1];
  v << sin(q[0][2]), -cos(q[0][2]);
  c[0] = x + r * v;

  // calculate the center of the circle to which q2 is tangent
  x << q[1][0], q[1][1];
  v << sin(q[1][2]), -cos(q[1][2]);
  c[1] = x + r * v;

  // the distance between the centers of these two circles
  d = (c[0] - c[1]).norm();

  // if the distance is too large, then this primitive is not the solution,
  // and we can bail here
  if (d > 4 * r) {
    for (int i = 0; i < 3; i++) l[i] = 0;
    feasible = false;
    return std::numeric_limits<double>::max();
  }

  feasible = true;

  if (d == 0) {
    Eigen::Map<Eigen::Vector3d>(l).fill(0);
    return 0;
  }

  // the base angle of the isosceles triangle whose vertices are the centers
  // of the the three circles, note acos returns [0,pi]
  a = acos(d / (4 * r));

  // create a clockwise rotation of magnitude alpha
  Rotation2Dd R(-a);

  // we find the third vertex of this triangle by taking the vector between
  // the two circle centers, normalizing it, and rotating it by alpha, and
  // scaling it to magnitude 2r, then it points from the center of
  // one the circle tangent to q1 to the third vertex
  c[2] = c[1] + R * (c[0] - c[1]).normalized() * 2 * r;

  // calculate the arc distance we travel on the first circle
  dc = c[2] - c[0];                       //< vector between centers of circles
  b = atan2(dc[1], dc[0]);                //< angle of that vector
  d = clampRadian(q[0][2] + M_PI / 2.0);  //< angle of vector from center to q1
  l[0] = clockwiseArc(d, b);              //< ccwise distance

  // calculate the arc distance we travel on the second circle
  dc = c[2] - c[1];                       //< vector between centers of circles
  b = atan2(dc[1], dc[0]);                //< angle of that vector
  d = clampRadian(q[1][2] + M_PI / 2.0);  //< angle of vector from center to q1
  l[1] = clockwiseArc(b, d);              //< ccwise distance

  // calculate the arc distance we travel on the third circle
  l[2] = M_PI - 2 * a;

  // sum the resulting segments
  double dist = 0;
  for (int i = 0; i < 3; i++) dist += l[i];

  return r * dist;
}

void SolverRLRb::draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  ctx->set_source(pat_R);

  drawCircle(ctx, c[0], r);
  ctx->stroke();

  drawCircle(ctx, c[1], r);
  ctx->stroke();

  if (feasible) {
    ctx->set_source(pat_L);
    drawCircle(ctx, c[2], r);
    ctx->stroke();
  }

  ctx->unset_dash();
  ctx->set_source(pat_q);

  if (!feasible) return;

  double a0, a1, a2;

  a0 = clampRadian(q[0][2] + M_PI / 2.0);
  a1 = clampRadian(q[1][2] + M_PI / 2.0);
  a2 = clampRadian(a0 - l[0] + M_PI);

  ctx->move_to(q[0][0], q[0][1]);
  ctx->arc_negative(c[0][0], c[0][1], r, a0, a0 - l[0]);
  ctx->arc(c[2][0], c[2][1], r, a2, a2 + l[2]);
  ctx->arc_negative(c[1][0], c[1][1], r, a1 + l[1], a1);

  ctx->stroke();
}

double SolverLSL::getDist(int i) {
  if (i == 2)
    return l[i];
  else
    return 180 * l[i] / M_PI;
}

double SolverLSL::solve(Eigen::Vector3d qin[2], double rin) {
  using namespace std;
  using namespace Eigen;

  for (int i = 0; i < 2; i++) q[i] = qin[i];
  r = rin;

  Vector2d x, v, dc;
  double a, b;

  // calculate the center of the circle to which q1 is tangent
  x << q[0][0], q[0][1];
  v << -sin(q[0][2]), cos(q[0][2]);
  c[0] = x + r * v;

  // calculate the center of the circle to which q2 is tangent
  x << q[1][0], q[1][1];
  v << -sin(q[1][2]), cos(q[1][2]);
  c[1] = x + r * v;

  // find the vector between the two
  v = c[1] - c[0];

  // the length of the straight segment is the length of this vector
  l[2] = v.norm();

  // rotate that vector -90 deg, normalize, and make it r in length
  a = M_PI / 2.0;

  // create a clockwise rotation of magnitude alpha
  Rotation2Dd R(-a);

  v = R * v.normalized() * r;

  // calculate the angle this vector makes with the circle
  a = atan2(v[1], v[0]);

  // now find the arc length for the two circles until it reaches this
  // point
  b = clampRadian(q[0][2] - M_PI / 2.0);
  l[0] = counterClockwiseArc(b, a);

  b = clampRadian(q[1][2] - M_PI / 2.0);
  l[1] = counterClockwiseArc(a, b);

  return r * (l[0] + l[1]) + l[2];
}

void SolverLSL::draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  ctx->set_source(pat_L);

  drawCircle(ctx, c[0], r);
  ctx->stroke();

  drawCircle(ctx, c[1], r);
  ctx->stroke();

  ctx->unset_dash();
  ctx->set_source(pat_q);

  double a0 = clampRadian(q[0][2] - M_PI / 2.0);
  double a1 = clampRadian(q[1][2] - M_PI / 2.0);

  ctx->move_to(q[0][0], q[0][1]);
  ctx->arc(c[0][0], c[0][1], r, a0, a0 + l[0]);

  Eigen::Vector2d v;
  v << r* cos(a1 - l[1]), r * sin(a1 - l[1]);
  v = c[1] + v;

  ctx->line_to(v[0], v[1]);
  ctx->arc(c[1][0], c[1][1], r, a1 - l[1], a1);

  ctx->stroke();
}

double SolverLSR::getDist(int i) {
  if (i == 2)
    return l[i];
  else
    return 180 * l[i] / M_PI;
}

double SolverLSR::solve(Eigen::Vector3d qin[2], double rin) {
  using namespace std;
  using namespace Eigen;

  for (int i = 0; i < 2; i++) q[i] = qin[i];
  r = rin;

  Vector2d x, v, n, dc, t1, t2;
  double a, b, d;

  // calculate the center of the circle to which q1 is tangent
  x << q[0][0], q[0][1];
  v << -sin(q[0][2]), cos(q[0][2]);
  c[0] = x + r * v;

  // calculate the center of the circle to which q2 is tangent
  x << q[1][0], q[1][1];
  v << sin(q[1][2]), -cos(q[1][2]);
  c[1] = x + r * v;

  // find the vector between the two
  v = c[1] - c[0];

  // calculate the distance between the two
  d = v.norm();

  // if they overlap then this primitive has no solution and is not the
  // optimal primitive
  if (d < 2 * r) {
    for (int i = 0; i < 3; i++) l[i] = 0;
    feasible = false;
    return std::numeric_limits<double>::max();
  }

  feasible = true;

  if (d == 0) {
    Eigen::Map<Eigen::Vector3d>(l).fill(0);
    return 0;
  }

  // find the angle between the line through centers and the radius that
  // points to the tangent on the circle
  a = acos(2 * r / d);

  // create a clockwise rotation of magnitude alpha
  Rotation2Dd R(-a);

  // get a normalized vector in this direction
  n = R * v.normalized();

  // get the tangent points
  t1 = c[0] + n * r;
  t2 = c[1] - n * r;

  // get the angle to the tangent points and the arclenghts on the two
  // circles
  a = atan2(n[1], n[0]);
  b = clampRadian(q[0][2] - M_PI / 2);
  l[0] = counterClockwiseArc(b, a);

  a = clampRadian(a + M_PI);
  b = clampRadian(q[1][2] + M_PI / 2);
  l[1] = clockwiseArc(a, b);

  // get the length of the segment
  l[2] = (t1 - t2).norm();

  return r * (l[0] + l[1]) + l[2];
}

void SolverLSR::draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  ctx->set_source(pat_L);
  drawCircle(ctx, c[0], r);
  ctx->stroke();

  ctx->set_source(pat_R);
  drawCircle(ctx, c[1], r);
  ctx->stroke();

  ctx->unset_dash();
  ctx->set_source(pat_q);

  if (!feasible) return;

  double a0 = clampRadian(q[0][2] - M_PI / 2.0);
  double a1 = clampRadian(q[1][2] + M_PI / 2.0);

  ctx->move_to(q[0][0], q[0][1]);
  ctx->arc(c[0][0], c[0][1], r, a0, a0 + l[0]);

  Eigen::Vector2d v;
  v << r* cos(a1 + l[1]), r * sin(a1 + l[1]);
  v = c[1] + v;

  ctx->line_to(v[0], v[1]);
  ctx->arc_negative(c[1][0], c[1][1], r, a1 + l[1], a1);

  ctx->stroke();
}

double SolverRSL::getDist(int i) {
  if (i == 2)
    return l[i];
  else
    return 180 * l[i] / M_PI;
}

double SolverRSL::solve(Eigen::Vector3d qin[2], double rin) {
  using namespace std;
  using namespace Eigen;

  for (int i = 0; i < 2; i++) q[i] = qin[i];
  r = rin;

  Vector2d x, v, n, dc, t1, t2;
  double a, b, d;

  // calculate the center of the circle to which q1 is tangent
  x << q[0][0], q[0][1];
  v << sin(q[0][2]), -cos(q[0][2]);
  c[0] = x + r * v;

  // calculate the center of the circle to which q2 is tangent
  x << q[1][0], q[1][1];
  v << -sin(q[1][2]), cos(q[1][2]);
  c[1] = x + r * v;

  // find the vector between the two
  v = c[1] - c[0];

  // calculate the distance between the two
  d = v.norm();

  // if they overlap then this primitive has no solution and is not the
  // optimal primitive
  if (d < 2 * r) {
    for (int i = 0; i < 3; i++) l[i] = 0;
    feasible = false;
    return std::numeric_limits<double>::max();
  }

  feasible = true;

  if (d == 0) {
    Eigen::Map<Eigen::Vector3d>(l).fill(0);
    return 0;
  }

  // find the angle between the line through centers and the radius that
  // points to the tangent on the circle
  a = acos(2 * r / d);

  // create a counter clockwise rotation of magnitude alpha
  Rotation2Dd R(a);

  // get a normalized vector in this direction
  n = R * v.normalized();

  // get the tangent points
  t1 = c[0] + n * r;
  t2 = c[1] - n * r;

  // get the angle to the tangent poitns and their arclengths
  a = atan2(n[1], n[0]);
  b = clampRadian(q[0][2] + M_PI / 2);
  l[0] = clockwiseArc(b, a);

  a = clampRadian(a + M_PI);
  b = clampRadian(q[1][2] - M_PI / 2);
  l[1] = counterClockwiseArc(a, b);

  // get the length of the segment
  l[2] = (t1 - t2).norm();

  return r * (l[0] + l[1]) + l[2];
}

void SolverRSL::draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  ctx->set_source(pat_R);
  drawCircle(ctx, c[0], r);
  ctx->stroke();

  ctx->set_source(pat_L);
  drawCircle(ctx, c[1], r);
  ctx->stroke();

  ctx->unset_dash();
  ctx->set_source(pat_q);

  if (!feasible) return;

  double a0 = clampRadian(q[0][2] + M_PI / 2.0);
  double a1 = clampRadian(q[1][2] - M_PI / 2.0);

  ctx->move_to(q[0][0], q[0][1]);
  ctx->arc_negative(c[0][0], c[0][1], r, a0, a0 - l[0]);

  Eigen::Vector2d v;
  v << r* cos(a1 - l[1]), r * sin(a1 - l[1]);
  v = c[1] + v;
  ctx->line_to(v[0], v[1]);

  ctx->arc(c[1][0], c[1][1], r, a1 - l[1], a1);

  ctx->stroke();
}

double SolverRSR::getDist(int i) {
  if (i == 2)
    return l[i];
  else
    return 180 * l[i] / M_PI;
}

double SolverRSR::solve(Eigen::Vector3d qin[2], double rin) {
  using namespace std;
  using namespace Eigen;

  for (int i = 0; i < 2; i++) q[i] = qin[i];
  r = rin;

  Vector2d x, v, dc;
  double a, b;

  // calculate the center of the circle to which q1 is tangent
  x << q[0][0], q[0][1];
  v << sin(q[0][2]), -cos(q[0][2]);
  c[0] = x + r * v;

  /// calculate the center of the circle to which q2 is tangent
  x << q[1][0], q[1][1];
  v << sin(q[1][2]), -cos(q[1][2]);
  c[1] = x + r * v;

  // find the vector between the two
  v = c[1] - c[0];

  // the length of the straight segment is the length of this vector
  l[2] = v.norm();

  // rotate that vector -90 deg, normalize, and make it r in length
  a = M_PI / 2.0;

  // create a clockwise rotation of magnitude alpha
  Rotation2Dd R(a);

  v = R * v.normalized() * r;

  // calculate the angle this vector makes with the circle
  a = atan2(v[1], v[0]);

  // now find the arc length for the two circles until it reaches this
  // point
  b = clampRadian(q[0][2] + M_PI / 2.0);
  l[0] = clockwiseArc(b, a);

  b = clampRadian(q[1][2] + M_PI / 2.0);
  l[1] = clockwiseArc(a, b);

  return r * (l[0] + l[1]) + l[2];
}

void SolverRSR::draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  ctx->set_source(pat_R);

  drawCircle(ctx, c[0], r);
  ctx->stroke();

  drawCircle(ctx, c[1], r);
  ctx->stroke();

  ctx->unset_dash();
  ctx->set_source(pat_q);

  double a0 = clampRadian(q[0][2] + M_PI / 2.0);
  double a1 = clampRadian(q[1][2] + M_PI / 2.0);

  ctx->move_to(q[0][0], q[0][1]);
  ctx->arc_negative(c[0][0], c[0][1], r, a0, a0 - l[0]);

  Eigen::Vector2d v;
  v << r* cos(a1 + l[1]), r * sin(a1 + l[1]);
  v = c[1] + v;

  ctx->line_to(v[0], v[1]);
  ctx->arc_negative(c[1][0], c[1][1], r, a1 + l[1], a1);

  ctx->stroke();
}

} // dubins
} // examples
} // mpblocks
