/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
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
 *  @date   Nov 5, 2012
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */

#ifndef MPBLOCKS_GTK_EIGEN_CAIRO_HPP_
#define MPBLOCKS_GTK_EIGEN_CAIRO_HPP_

#include <mpblocks/gtk/EigenCairo.h>

namespace mpblocks {
namespace      gtk {

template <typename Derived>
void EigenCairo::translate(const Eigen::MatrixBase<Derived>& v) {
  m_ctx->translate((double)v[0],
                   (double)v[1]);
}

template <typename Derived>
void EigenCairo::scale(const Eigen::MatrixBase<Derived>& v) {
  m_ctx->scale((double)v[0],
               (double)v[1]);
}

template <typename Derived>
void EigenCairo::user_to_device(Eigen::MatrixBase<Derived>& v) const {
  double x = v[0];
  double y = v[1];
  m_ctx->user_to_device(x, y);
  v[0] = x;
  v[1] = y;
}

template <typename Derived>
void EigenCairo::move_to(const Eigen::MatrixBase<Derived>& v) const {
  m_ctx->move_to((double)v[0],
                 (double)v[1]);
}

template <typename Derived>
void EigenCairo::line_to(const Eigen::MatrixBase<Derived>& v) const {
  m_ctx->line_to((double)v[0],
                 (double)v[1]);
}

template <typename Derived1, typename Derived2, typename Derived3>
void EigenCairo::curve_to(const Eigen::MatrixBase<Derived1>& x,
                          const Eigen::MatrixBase<Derived2>& c1,
                          const Eigen::MatrixBase<Derived3>& c2) {
  m_ctx->curve_to((double)x[0], (double)x[1],
                  (double)c1[0], (double)c1[1],
                  (double)c2[0], (double)c2[1]);
}

template <typename Derived>
void EigenCairo::arc(const Eigen::MatrixBase<Derived>& c, double radius,
                     double angle1, double angle2) {
  m_ctx->arc((double)c[0], (double)c[1], radius, angle1, angle2);
}

template <typename Derived>
void EigenCairo::arc_negative(const Eigen::MatrixBase<Derived>& c,
                              double radius, double angle1, double angle2) {
  m_ctx->arc_negative((double)c[0], (double)c[1], radius, angle1, angle2);
}

template <typename Derived>
void EigenCairo::rel_move_to(const Eigen::MatrixBase<Derived>& v) {
  m_ctx->rel_move_to((double)v[0], (double)v[1]);
}

template <typename Derived>
void EigenCairo::rel_line_to(const Eigen::MatrixBase<Derived>& v) {
  m_ctx->rel_line_to((double)v[0], (double)v[1]);
}

template <typename Derived1, typename Derived2, typename Derived3>
void EigenCairo::rel_curve_to(const Eigen::MatrixBase<Derived1>& dx,
                              const Eigen::MatrixBase<Derived2>& dc1,
                              const Eigen::MatrixBase<Derived3>& dc2) {
  m_ctx->rel_curve_to((double)dx[0], (double)dx[1],
                      (double)dc1[0], (double)dc1[1],
                      (double)dc2[0], (double)dc2[1]);
}

template <typename Derived1, typename Derived2>
void EigenCairo::rectangle(const Eigen::MatrixBase<Derived1>& x,
                           const Eigen::MatrixBase<Derived2>& s) {
  m_ctx->rectangle((double)x[0], (double)x[1],
                   (double)s[0], (double)s[1]);
}

template <typename Derived>
inline void EigenCairo::circle(const Eigen::MatrixBase<Derived>& c,
                               double r) {
  m_ctx->move_to((double)c[0] + r, (double)c[1]);
  this->arc(c, r, 0, 2 * M_PI);
}

}  // namespace gtk
}  // namespace mpblocks

#endif // MPBLOCKS_GTK_EIGEN_CAIRO_HPP_
