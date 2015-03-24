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
 *
 *  @date   Jun 30, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_GENERICARTIST_HPP_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_GENERICARTIST_HPP_

#include <mpblocks/gtk.h>

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {

/// interface for different solutions
template <typename Format_t>
struct GenericArtist {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef Eigen::Matrix<Format_t, 2, 1> Vector2d_t;
  typedef Integrate<Format_t>           Integrate_t;
  typedef Path<Format_t>                Path_t;

  static void drawLRL(const Vector3d_t& q0, const Format_t r, const Path_t soln,
                      const DrawOpts& opts) {
    if (!soln.f) return;

    Cairo::RefPtr<Cairo::Context> ctx = opts.ctx;
    gtk::EigenCairo ectx(ctx);

    Vector3d_t q1 = Integrate_t::L(q0, r, soln.s[0]);
    Vector3d_t q2 = Integrate_t::R(q1, r, soln.s[1]);

    Vector2d_t c[3];
    c[0] = leftCenter(q0, r);
    c[1] = rightCenter(q1, r);
    c[2] = leftCenter(q2, r);

    Vector3d_t alpha;
    alpha[0] = leftAngleOf(q0);
    alpha[1] = rightAngleOf(q1);
    alpha[2] = leftAngleOf(q2);

    Vector3d_t arc = soln.s;

    if (opts.drawBalls) {
      ctx->set_dash(opts.dash, opts.dashOffExtra);
      ctx->set_source(opts.patL);
      ectx.circle(c[0], r);
      ctx->stroke();
      ectx.circle(c[2], r);
      ctx->stroke();
      ctx->set_source(opts.patR);
      ectx.circle(c[1], r);
      ctx->stroke();
      ctx->unset_dash();
    }

    ctx->set_source(opts.patPath);
    ctx->move_to(q0[0], q0[1]);
    ectx.arc( c[0], r, alpha[0], alpha[0] + arc[0]);
    ectx.arc_negative(c[1], r, alpha[1], alpha[1] - arc[1]);
    ectx.arc(c[2], r, alpha[2], alpha[2] + arc[2]);
    ctx->stroke();
  }

  static void drawRLR(const Vector3d_t& q0, const Format_t r,
                      const Path_t& soln, const DrawOpts& opts) {
    if (!soln.f) return;

    Cairo::RefPtr<Cairo::Context> ctx = opts.ctx;
    gtk::EigenCairo ectx(ctx);

    Vector3d_t q1 = Integrate_t::R(q0, r, soln.s[0]);
    Vector3d_t q2 = Integrate_t::L(q1, r, soln.s[1]);

    Vector2d_t c[3];
    c[0] = rightCenter(q0, r);
    c[1] = leftCenter(q1, r);
    c[2] = rightCenter(q2, r);

    Vector3d_t alpha;
    alpha[0] = rightAngleOf(q0);
    alpha[1] = leftAngleOf(q1);
    alpha[2] = rightAngleOf(q2);

    Vector3d_t arc = soln.s;

    if (opts.drawBalls) {
      ctx->set_dash(opts.dash, opts.dashOffExtra);
      ctx->set_source(opts.patR);
      ectx.circle(c[0], r);
      ctx->stroke();
      ectx.circle(c[2], r);
      ctx->stroke();
      ctx->set_source(opts.patL);
      ectx.circle(c[1], r);
      ctx->stroke();
      ctx->unset_dash();
    }

    ctx->set_source(opts.patPath);
    ctx->move_to(q0[0], q0[1]);
    ectx.arc_negative(c[0], r, alpha[0], alpha[0] - arc[0]);
    ectx.arc(c[1], r, alpha[1], alpha[1] + arc[1]);
    ectx.arc_negative(c[2], r, alpha[2], alpha[2] - arc[2]);
    ctx->stroke();
  }

  static void drawLSL(const Vector3d_t& q0, const Format_t r,
                      const Path_t& soln, const DrawOpts& opts) {
    if (!soln.f) return;

    Cairo::RefPtr<Cairo::Context> ctx = opts.ctx;
    gtk::EigenCairo ectx(ctx);

    Vector3d_t q1 = Integrate_t::L(q0, r, soln.s[0]);
    Vector3d_t q2 = Integrate_t::S(q1, soln.s[1]);
    Vector3d_t q3 = Integrate_t::L(q2, r, soln.s[2]);

    Vector2d_t c[2];
    c[0] = leftCenter(q0, r);
    c[1] = leftCenter(q3, r);

    Format_t alpha[2];
    alpha[0] = leftAngleOf(q0);
    alpha[1] = leftAngleOf(q2);

    if (opts.drawBalls) {
      ctx->set_dash(opts.dash, opts.dashOffExtra);
      ctx->set_source(opts.patL);
      ectx.circle(c[0], r);
      ctx->stroke();
      ectx.circle(c[1], r);
      ctx->stroke();
      ctx->unset_dash();
    }

    // get the angle of the straight segment
    Format_t angle = leftAngle_inv(alpha[0] + soln.s[0]);

    // create a vector
    Vector2d_t v = soln.s[1] * Vector2d_t(std::cos(angle), std::sin(angle));

    ctx->set_source(opts.patPath);
    ctx->move_to(q0[0], q0[1]);
    ectx.arc(c[0], r, alpha[0], alpha[0] + soln.s[0]);
    ectx.rel_line_to(v);
    ectx.arc(c[1], r, alpha[1], alpha[1] + soln.s[2]);
    ctx->stroke();
  }

  static void drawRSR(const Vector3d_t& q0, const Format_t r,
                      const Path_t& soln, const DrawOpts& opts) {
    if (!soln.f) return;

    Cairo::RefPtr<Cairo::Context> ctx = opts.ctx;
    gtk::EigenCairo ectx(ctx);

    Vector3d_t q1 = Integrate_t::R(q0, r, soln.s[0]);
    Vector3d_t q2 = Integrate_t::S(q1, soln.s[1]);

    Vector2d_t c[2];
    c[0] = rightCenter(q0, r);
    c[1] = rightCenter(q2, r);

    Vector3d_t alpha;
    alpha[0] = rightAngleOf(q0);
    alpha[1] = rightAngleOf(q2);

    if (opts.drawBalls) {
      ctx->set_dash(opts.dash, opts.dashOffExtra);
      ctx->set_source(opts.patR);
      ectx.circle(c[0], r);
      ctx->stroke();
      ectx.circle(c[1], r);
      ctx->stroke();
      ctx->unset_dash();
    }

    // get the angle of the straight segment
    Format_t angle = rightAngle_inv(alpha[0] - soln.s[0]);

    // create a vector
    Vector2d_t v = soln.s[1] * Vector2d_t(std::cos(angle), std::sin(angle));

    ctx->set_source(opts.patPath);
    ctx->move_to(q0[0], q0[1]);
    ectx.arc_negative(c[0], r, alpha[0], alpha[0] - soln.s[0]);
    ectx.rel_line_to(v);
    ectx.arc_negative(c[1], r, alpha[1], alpha[1] - soln.s[2]);
    ctx->stroke();
  }

  static void drawLSR(const Vector3d_t& q0, const Format_t r,
                      const Path_t& soln, const DrawOpts& opts) {
    if (!soln.f) return;

    Cairo::RefPtr<Cairo::Context> ctx = opts.ctx;
    gtk::EigenCairo ectx(ctx);

    Vector3d_t q1 = Integrate_t::L(q0, r, soln.s[0]);
    Vector3d_t q2 = Integrate_t::S(q1, soln.s[1]);

    Vector2d_t c[2];
    c[0] = leftCenter(q0, r);
    c[1] = rightCenter(q2, r);

    Vector3d_t alpha;
    alpha[0] = leftAngleOf(q0);
    alpha[1] = rightAngleOf(q2);

    if (opts.drawBalls) {
      ctx->set_dash(opts.dash, opts.dashOffExtra);
      ctx->set_source(opts.patL);
      ectx.circle(c[0], r);
      ctx->stroke();
      ctx->set_source(opts.patR);
      ectx.circle(c[1], r);
      ctx->stroke();
      ctx->unset_dash();
    }

    // get the angle of the straight segment
    Format_t angle = leftAngle_inv(alpha[0] + soln.s[0]);

    // create a vector
    Vector2d_t v = soln.s[1] * Vector2d_t(std::cos(angle), std::sin(angle));

    ctx->set_source(opts.patPath);
    ctx->move_to(q0[0], q0[1]);
    ectx.arc(c[0], r, alpha[0], alpha[0] + soln.s[0]);
    ectx.rel_line_to(v);
    ectx.arc_negative(c[1], r, alpha[1], alpha[1] - soln.s[2]);
    ctx->stroke();
  }

  static void drawRSL(const Vector3d_t& q0, const Format_t r,
                      const Path_t& soln, const DrawOpts& opts) {
    if (!soln.f) return;

    Cairo::RefPtr<Cairo::Context> ctx = opts.ctx;
    gtk::EigenCairo ectx(ctx);

    Vector3d_t q1 = Integrate_t::R(q0, r, soln.s[0]);
    Vector3d_t q2 = Integrate_t::S(q1, soln.s[1]);

    Vector2d_t c[2];
    c[0] = rightCenter(q0, r);
    c[1] = leftCenter(q2, r);

    Vector3d_t alpha;
    alpha[0] = rightAngleOf(q0);
    alpha[1] = leftAngleOf(q2);

    if (opts.drawBalls) {
      ctx->set_dash(opts.dash, opts.dashOffExtra);
      ctx->set_source(opts.patR);
      ectx.circle(c[0], r);
      ctx->stroke();
      ctx->set_source(opts.patL);
      ectx.circle(c[1], r);
      ctx->stroke();
      ctx->unset_dash();
    }

    // get the angle of the straight segment
    Format_t angle = rightAngle_inv(alpha[0] - soln.s[0]);

    // create a vector
    Vector2d_t v = soln.s[1] * Vector2d_t(std::cos(angle), std::sin(angle));

    ctx->set_source(opts.patPath);
    ctx->move_to(q0[0], q0[1]);
    ectx.arc_negative(c[0], r, alpha[0], alpha[0] - soln.s[0]);
    ectx.rel_line_to(v);
    ectx.arc(c[1], r, alpha[1], alpha[1] + soln.s[2]);
    ctx->stroke();
  }

  static void drawLS(const Vector3d_t& q0, const Format_t r, const Path_t& soln,
                     const DrawOpts& opts) {
    if (!soln.f) return;

    Cairo::RefPtr<Cairo::Context> ctx = opts.ctx;
    gtk::EigenCairo ectx(ctx);

    Vector3d_t q1 = Integrate_t::L(q0, r, soln.s[0]);
    Vector3d_t q2 = Integrate_t::S(q1, soln.s[1]);

    Vector2d_t c = leftCenter(q0, r);

    if (opts.drawBalls) {
      ctx->set_dash(opts.dash, opts.dashOffExtra);
      ctx->set_source(opts.patL);
      ectx.circle(c, r);
      ctx->stroke();
      ctx->unset_dash();
    }

    ctx->set_source(opts.patPath);

    Format_t a1 = leftAngleOf(q0);
    ctx->move_to(q0[0], q0[1]);
    ectx.arc(c, r, a1, a1 + soln.s[0]);
    Vector2d_t v = soln.s[1] * Vector2d_t(std::cos(q1[2]), std::sin(q1[2]));
    ectx.rel_line_to(v);
    ctx->stroke();
  }

  static void drawRS(const Vector3d_t& q0, const Format_t r, const Path_t& soln,
                     const DrawOpts& opts) {
    if (!soln.f) return;

    Cairo::RefPtr<Cairo::Context> ctx = opts.ctx;
    gtk::EigenCairo ectx(ctx);

    Vector3d_t q1 = Integrate_t::R(q0, r, soln.s[0]);
    Vector3d_t q2 = Integrate_t::S(q1, soln.s[1]);

    Vector2d_t c = rightCenter(q0, r);

    if (opts.drawBalls) {
      ctx->set_dash(opts.dash, opts.dashOffExtra);
      ctx->set_source(opts.patR);
      ectx.circle(c, r);
      ctx->stroke();
      ctx->unset_dash();
    }

    ctx->set_source(opts.patPath);

    Format_t a1 = rightAngleOf(q0);
    ctx->move_to(q0[0], q0[1]);
    ectx.arc_negative(c, r, a1, a1 - soln.s[0]);

    Vector2d_t v = soln.d1 * Vector2d_t(std::cos(q1[2]), std::sin(q1[2]));
    ectx.rel_line_to(v);
    ctx->stroke();
  }

  /// dispatch the artist
  static void draw(const Vector3d_t& q0, const Format_t r, const Path_t& soln,
                   const DrawOpts& opts) {
    switch (soln.id) {
      case LRLa:
      case LRLb:
        drawLRL(q0, r, soln, opts);
        return;

      case RLRa:
      case RLRb:
        drawRLR(q0, r, soln, opts);
        return;

      case LSL:
        drawLSL(q0, r, soln, opts);
        return;

      case RSR:
        drawRSR(q0, r, soln, opts);
        return;

      case LSR:
        drawLSR(q0, r, soln, opts);
        return;

      case RSL:
        drawRSL(q0, r, soln, opts);
        return;
    }
  }
};

template <typename Format_t>
void draw(const Eigen::Matrix<Format_t, 3, 1>& q0, const Format_t r,
          const Path<Format_t>& soln, const DrawOpts& opts) {
  GenericArtist<Format_t>::draw(q0, r, soln, opts);
}

}  // curves_eigen
}  // dubins
}  // mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_GENERICARTIST_HPP_
