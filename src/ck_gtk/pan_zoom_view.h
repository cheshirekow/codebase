/*
 *  Copyright (C) 2014 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of ck_gtk.
 *
 *  ck_gtk is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  ck_gtk is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with ck_gtk.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CK_GTK_PAN_ZOOM_VIEW_H_
#define CK_GTK_PAN_ZOOM_VIEW_H_

#include <Eigen/Dense>
#include <gtkmm.h>

namespace ck_gtk {

/// A drawing area with built in mouse handlers for pan-zoom control
class PanZoomView : public Gtk::DrawingArea {
 private:
  Glib::RefPtr<Gtk::Adjustment> offset_x_;  ///< offset of viewport
  Glib::RefPtr<Gtk::Adjustment> offset_y_;  ///< offset of viewport

  /// Size (in virtual units) of the largest dimension of the widget
  Glib::RefPtr<Gtk::Adjustment> scale_;

  /// When the mouse wheel is turned multiply or divide the scale by this
  /// much
  Glib::RefPtr<Gtk::Adjustment> scale_rate_;

  /// The last mouse position (used during pan)
  Eigen::Vector2d last_pos_;

  /// If true, controls are enabled, if false, controls are disabled and
  /// events are passed through
  bool active_;

  /// Which mouse button is used for pan
  int pan_button_;
  guint pan_button_mask_;

 public:
  /// Signal is emitted by the on_draw handler, and sends out the context
  /// with appropriate scaling and
  sigc::signal<void, const Cairo::RefPtr<Cairo::Context>&> sig_draw;

  /// motion event with transformed coordinates
  sigc::signal<void, GdkEventMotion*> sig_motion;

  /// button event with transformed coordinates
  sigc::signal<void, GdkEventButton*> sig_button;

  PanZoomView();

  void SetOffsetAdjustments(Glib::RefPtr<Gtk::Adjustment> offset_x,
                            Glib::RefPtr<Gtk::Adjustment> offset_y);

  void SetScaleAdjustments(Glib::RefPtr<Gtk::Adjustment> scale,
                           Glib::RefPtr<Gtk::Adjustment> scale_rate);

  void SetOffset(const Eigen::Vector2d& offset);

  Eigen::Vector2d GetOffset();

  void SetScale(double scale);

  double GetScale();
  void SetScaleRate(double scale_rate);

  double GetScaleRate();

  /// return the maximum dimension of the
  double GetMaxDim();

  /// Convert the point (x,y) in GTK coordinates, with the origin at the top
  /// left, to a point in traditional cartesian coordinates, where the origin
  /// is at the bottom left.
  Eigen::Vector2d RawPoint(double x, double y);

  /// Return a point in the virtual cartesian plane by applying the offset
  /// and scaling of the viewport
  Eigen::Vector2d TransformPoint(double x, double y);

  template <typename Event>
  void TransformEvent(Event* event) {
    Eigen::Vector2d transformed_point = TransformPoint(event->x, event->y);
    event->x = transformed_point[0];
    event->y = transformed_point[1];
  }

  // overrides base class handlers
  virtual bool on_motion_notify_event(GdkEventMotion* event);
  virtual bool on_button_press_event(GdkEventButton* event);
  virtual bool on_scroll_event(GdkEventScroll* event);
  virtual bool on_draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

}  // namespace ck_gtk

#endif  // CK_GTK_PAN_ZOOM_VIEW_H_
