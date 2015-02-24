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

#ifndef MPBLOCKS_GTK_EIGEN_CAIRO_H_
#define MPBLOCKS_GTK_EIGEN_CAIRO_H_

#include <gtkmm.h>
#include <cairomm/cairomm.h>
#include <Eigen/Dense>

namespace mpblocks {
namespace      gtk {

/// wrapper for cairo context which adds methods allowing for eigen
/// vector typed points
class EigenCairo {
 private:
  Cairo::RefPtr<Cairo::Context> m_ctx;

 public:
  EigenCairo(const Cairo::RefPtr<Cairo::Context>& ctx);

  /** Sets the source pattern within the Context to source. This Pattern will
    * then be used for any subsequent drawing operation until a new source
    * pattern is set.
    *
    * Note: The Pattern's transformation matrix will be locked to the user space
    * in effect at the time of set_source(). This means that further
    * modifications of the current transformation matrix will not affect the
    * source pattern.
    *
    * @param source  a Pattern to be used as the source for subsequent drawing
    * operations.
    *
    * @sa Pattern::set_matrix()
    * @sa set_source_rgb()
    * @sa set_source_rgba()
    * @sa set_source(const RefPtr<Surface>& surface, double x, double y)
    */
  void set_source(const Gdk::RGBA& rgba);

  /** Sets the source pattern within the Context to an opaque color. This
    * opaque color will then be used for any subsequent drawing operation until
    * a new source pattern is set.
    *
    * The color components are floating point numbers in the range 0 to 1. If
    * the values passed in are outside that range, they will be clamped.
    *
    * @param red red component of color
    * @param green   green component of color
    * @param blue    blue component of color
    *
    * @sa set_source_rgba()
    * @sa set_source()
    */
  void set_source_rgb(const char* str);

  /** Sets the source pattern within the Context to a translucent color. This
    * color will then be used for any subsequent drawing operation until a new
    * source pattern is set.
    *
    * The color and alpha components are floating point numbers in the range 0
    * to 1. If the values passed in are outside that range, they will be
    * clamped.
    *
    * @param red red component of color
    * @param green   green component of color
    * @param blue    blue component of color
    * @param alpha   alpha component of color
    *
    * @sa set_source_rgb()
    * @sa set_source()
    */
  void set_source_rgba(const char* str);

  /** Modifies the current transformation matrix (CTM) by translating the
    * user-space origin by (tx, ty). This offset is interpreted as a user-space
    * coordinate according to the CTM in place before the new call to
    * cairo_translate. In other words, the translation of the user-space origin
    * takes place after any existing transformation.
    *
    * @param tx  amount to translate in the X direction
    * @param ty  amount to translate in the Y direction
    */
  template <typename Derived>
  void translate(const Eigen::MatrixBase<Derived>& v);

  /** Modifies the current transformation matrix (CTM) by scaling the X and Y
    * user-space axes by sx and sy respectively. The scaling of the axes takes
    * place after any existing transformation of user space.
    *
    * @param sx  scale factor for the X dimension
    * @param sy  scale factor for the Y dimension
    */
  template <typename Derived>
  void scale(const Eigen::MatrixBase<Derived>& v);

  /** Transform a coordinate from user space to device space by multiplying the
    * given point by the current transformation matrix (CTM).
    *
    * @param x   X value of coordinate (in/out parameter)
    * @param y   Y value of coordinate (in/out parameter)
    */
  template <typename Derived>
  void user_to_device(Eigen::MatrixBase<Derived>& v) const;

  void user_to_device(Eigen::Matrix<double, 2, 1>& v) const;

  /** If the current subpath is not empty, begin a new subpath. After this call
    * the current point will be (x, y).
    *
    * @param x   the X coordinate of the new position
    * @param y   the Y coordinate of the new position
    */
  template <typename Derived>
  void move_to(const Eigen::MatrixBase<Derived>& v) const;

  /** Adds a line to the path from the current point to position (x, y) in
    * user-space coordinates. After this call the current point will be (x, y).
    *
    * @param x   the X coordinate of the end of the new line
    * @param y   the Y coordinate of the end of the new line
    */
  template <typename Derived>
  void line_to(const Eigen::MatrixBase<Derived>& v) const;

  /** Adds a cubic Bezier spline to the path from the current point to position
    * (x3, y3) in user-space coordinates, using (x1, y1) and (x2, y2) as the
    * control points. After this call the current point will be (x3, y3).
    *
    * @param x1  the X coordinate of the first control point
    * @param y1  the Y coordinate of the first control point
    * @param x2  the X coordinate of the second control point
    * @param y2  the Y coordinate of the second control point
    * @param x3  the X coordinate of the end of the curve
    * @param y3  the Y coordinate of the end of the curve
    */
  template <typename Derived1, typename Derived2, typename Derived3>
  void curve_to(const Eigen::MatrixBase<Derived1>& x,
                const Eigen::MatrixBase<Derived2>& c1,
                const Eigen::MatrixBase<Derived3>& c2);

  /** Adds a circular arc of the given radius to the current path. The arc is
    * centered at (@a xc, @a yc), begins at @a angle1 and proceeds in the
    *direction of
    * increasing angles to end at @a angle2. If @a angle2 is less than @a angle1
    *it will
    * be progressively increased by 2*M_PI until it is greater than @a angle1.
    *
    * If there is a current point, an initial line segment will be added to the
    * path to connect the current point to the beginning of the arc. If this
    * initial line is undesired, it can be avoided by calling
    * begin_new_sub_path() before calling arc().
    *
    * Angles are measured in radians. An angle of 0 is in the direction of the
    * positive X axis (in user-space). An angle of M_PI/2.0 radians (90 degrees)
    *is
    * in the direction of the positive Y axis (in user-space). Angles increase
    * in the direction from the positive X axis toward the positive Y axis. So
    * with the default transformation matrix, angles increase in a clockwise
    * direction.
    *
    * ( To convert from degrees to radians, use degrees * (M_PI / 180.0). )
    *
    * This function gives the arc in the direction of increasing angles; see
    * arc_negative() to get the arc in the direction of decreasing angles.
    *
    * The arc is circular in user-space. To achieve an elliptical arc, you can
    * scale the current transformation matrix by different amounts in the X and
    * Y directions. For example, to draw an ellipse in the box given by x, y,
    * width, height:
    *
    * @code
    * context->save();
    * context->translate(x, y);
    * context->scale(width / 2.0, height / 2.0);
    * context->arc(0.0, 0.0, 1.0, 0.0, 2 * M_PI);
    * context->restore();
    * @endcode
    *
    * @param xc  X position of the center of the arc
    * @param yc  Y position of the center of the arc
    * @param radius  the radius of the arc
    * @param angle1  the start angle, in radians
    * @param angle2  the end angle, in radians
    */
  template <typename Derived>
  void arc(const Eigen::MatrixBase<Derived>& c, double radius, double angle1,
           double angle2);

  /** Adds a circular arc of the given @a radius to the current path. The arc is
    * centered at (@a xc, @a yc), begins at @a angle1 and proceeds in the
    *direction of
    * decreasing angles to end at @a angle2. If @a angle2 is greater than @a
    *angle1 it
    * will be progressively decreased by 2*M_PI until it is greater than @a
    *angle1.
    *
    * See arc() for more details. This function differs only in the direction of
    * the arc between the two angles.
    *
    * @param xc  X position of the center of the arc
    * @param yc  Y position of the center of the arc
    * @param radius  the radius of the arc
    * @param angle1  the start angle, in radians
    * @param angle2  the end angle, in radians
    */
  template <typename Derived>
  void arc_negative(const Eigen::MatrixBase<Derived>& c, double radius,
                    double angle1, double angle2);

  /** If the current subpath is not empty, begin a new subpath. After this call
    * the current point will offset by (x, y).
    *
    * Given a current point of (x, y),
    * @code
    * rel_move_to(dx, dy)
    * @endcode
    * is logically equivalent to
    * @code
    * move_to(x + dx, y + dy)
    * @endcode
    *
    * @param dx  the X offset
    * @param dy  the Y offset
    *
    * It is an error to call this function with no current point. Doing
    * so will cause this to shutdown with a status of
    * CAIRO_STATUS_NO_CURRENT_POINT. Cairomm will then throw an exception.
    */
  template <typename Derived>
  void rel_move_to(const Eigen::MatrixBase<Derived>& v);

  /** Relative-coordinate version of line_to(). Adds a line to the path from
    * the current point to a point that is offset from the current point by (dx,
    * dy) in user space. After this call the current point will be offset by
    * (dx, dy).
    *
    * Given a current point of (x, y),
    * @code
    * rel_line_to(dx, dy)
    * @endcode
    * is logically equivalent to
    * @code
    * line_to(x + dx, y + dy).
    * @endcode
    *
    * @param dx  the X offset to the end of the new line
    * @param dy  the Y offset to the end of the new line
    *
    * It is an error to call this function with no current point. Doing
    * so will cause this to shutdown with a status of
    * CAIRO_STATUS_NO_CURRENT_POINT. Cairomm will then throw an exception.
    */
  template <typename Derived>
  void rel_line_to(const Eigen::MatrixBase<Derived>& v);

  /** Relative-coordinate version of curve_to(). All offsets are relative to
    * the current point. Adds a cubic Bezier spline to the path from the current
    * point to a point offset from the current point by (dx3, dy3), using points
    * offset by (dx1, dy1) and (dx2, dy2) as the control points.  After this
    * call the current point will be offset by (dx3, dy3).
    *
    * Given a current point of (x, y),
    * @code
    * rel_curve_to(dx1, dy1, dx2, dy2, dx3, dy3)
    * @endcode
    * is logically equivalent to
    * @code
    * curve_to(x + dx1, y + dy1, x + dx2, y + dy2, x + dx3, y + dy3).
    * @endcode
    *
    * @param dx1 the X offset to the first control point
    * @param dy1 the Y offset to the first control point
    * @param dx2 the X offset to the second control point
    * @param dy2 the Y offset to the second control point
    * @param dx3 the X offset to the end of the curve
    * @param dy3 the Y offset to the end of the curve
    *
    * It is an error to call this function with no current point. Doing
    * so will cause this to shutdown with a status of
    * CAIRO_STATUS_NO_CURRENT_POINT. Cairomm will then throw an exception.
    */
  template <typename Derived1, typename Derived2, typename Derived3>
  void rel_curve_to(const Eigen::MatrixBase<Derived1>& dx,
                    const Eigen::MatrixBase<Derived2>& dc1,
                    const Eigen::MatrixBase<Derived3>& dc2);

  /** Adds a closed-subpath rectangle of the given size to the current path at
    * position (x, y) in user-space coordinates.
    *
    * This function is logically equivalent to:
    *
    * @code
    * context->move_to(x, y);
    * context->rel_line_to(width, 0);
    * context->rel_line_to(0, height);
    * context->rel_line_to(-width, 0);
    * context->close_path();
    * @endcode
    *
    * @param x   the X coordinate of the top left corner of the rectangle
    * @param y   the Y coordinate to the top left corner of the rectangle
    * @param width   the width of the rectangle
    * @param height  the height of the rectangle
    */
  template <typename Derived1, typename Derived2>
  void rectangle(const Eigen::MatrixBase<Derived1>& x,
                 const Eigen::MatrixBase<Derived2>& s);

  template <typename Derived>
  void circle(const Eigen::MatrixBase<Derived>& c, double r);

  struct SolidPattern {
    /// create a solid pattern from a CSS string
    static Cairo::RefPtr<Cairo::SolidPattern> create_rgb(const char* str);

    /// create a solid pattern from a CSS string
    static Cairo::RefPtr<Cairo::SolidPattern> create_rgba(const char* str);

    /// create a solid pattern from a CSS string
    static Cairo::RefPtr<Cairo::SolidPattern> create_rgb(const Gdk::Color& rgb);

    /// create a solid pattern from a CSS string
    static Cairo::RefPtr<Cairo::SolidPattern> create_rgba(
        const Gdk::RGBA& rgba);
  };
};

} // namespace gtk
} // namespace mpblocks



#endif // CAIRO_H_
