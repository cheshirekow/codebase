/*
 *  Copyright (C) 2014 Josh Bialkowski (josh.bialkowski@gmail.com)
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
 *  @date   Nov 19, 2014
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */
#ifndef MPBLOCKS_GTK_PAN_ZOOM_VIEW_H_
#define MPBLOCKS_GTK_PAN_ZOOM_VIEW_H_

#include <Eigen/Dense>
#include <gtkmm.h>

namespace mpblocks {
namespace      gtk {

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

  PanZoomView() {
    add_events(Gdk::POINTER_MOTION_MASK | Gdk::BUTTON_MOTION_MASK |
               Gdk::BUTTON_PRESS_MASK | Gdk::BUTTON_RELEASE_MASK |
               Gdk::SCROLL_MASK);
    offset_x_ = Gtk::Adjustment::create(0, -1e9, 1e9);
    offset_y_ = Gtk::Adjustment::create(0, -1e9, 1e9);
    scale_ = Gtk::Adjustment::create(1, 1e-9, 1e9);
    scale_rate_ = Gtk::Adjustment::create(1.05, 0, 10);
    active_ = true;
    pan_button_ = 3;
    pan_button_mask_ = GDK_BUTTON3_MASK;
  }

  void SetOffsetAdjustments(Glib::RefPtr<Gtk::Adjustment> offset_x,
                            Glib::RefPtr<Gtk::Adjustment> offset_y) {
    offset_x_ = offset_x;
    offset_y_ = offset_y;
    offset_x_->signal_value_changed().connect(
        sigc::mem_fun(this, &Gtk::DrawingArea::queue_draw));
    offset_y_->signal_value_changed().connect(
        sigc::mem_fun(this, &Gtk::DrawingArea::queue_draw));
  }

  void SetScaleAdjustments(Glib::RefPtr<Gtk::Adjustment> scale,
                           Glib::RefPtr<Gtk::Adjustment> scale_rate) {
    scale_ = scale;
    scale_rate_ = scale_rate;
    scale_->signal_value_changed().connect(
        sigc::mem_fun(this, &Gtk::DrawingArea::queue_draw));
  }

  void SetOffset(const Eigen::Vector2d& offset) {
    if (offset_x_) {
      offset_x_->set_value(offset[0]);
    }
    if (offset_y_) {
      offset_y_->set_value(offset[1]);
    }
  }

  Eigen::Vector2d GetOffset() {
    return (offset_x_ && offset_y_)
               ? Eigen::Vector2d(offset_x_->get_value(), offset_y_->get_value())
               : Eigen::Vector2d(0, 0);
  }

  void SetScale(double scale) {
    if (scale_) {
      scale_->set_value(scale);
    }
  }

  double GetScale() { return scale_ ? scale_->get_value() : 1; }

  void SetScaleRate(double scale_rate) {
    if (scale_rate_) {
      scale_rate_->set_value(scale_rate);
    }
  }

  double GetScaleRate() {
    return scale_rate_ ? scale_rate_->get_value() : 1.05;
  }

  /// return the maximum dimension of the
  double GetMaxDim() {
    return std::max(get_allocated_width(), get_allocated_height());
  }

  /// Convert the point (x,y) in GTK coordinates, with the origin at the top
  /// left, to a point in traditional cartesian coordinates, where the origin
  /// is at the bottom left.
  Eigen::Vector2d RawPoint(double x, double y) {
    return Eigen::Vector2d(x, get_allocated_height() - y);
  }

  /// Return a point in the virtual cartesian plane by applying the offset
  /// and scaling of the viewport
  Eigen::Vector2d TransformPoint(double x, double y) {
    return GetOffset() + RawPoint(x, y) * (GetScale() / GetMaxDim());
  }

  template <typename Event>
  void TransformEvent(Event* event) {
    Eigen::Vector2d transformed_point = TransformPoint(event->x, event->y);
    event->x = transformed_point[0];
    event->y = transformed_point[1];
  }

  virtual bool on_motion_notify_event(GdkEventMotion* event) {
    if (active_ && (event->state & pan_button_mask_)) {
      Eigen::Vector2d loc = RawPoint(event->x, event->y);
      Eigen::Vector2d delta = loc - last_pos_;
      Eigen::Vector2d new_offset =
          GetOffset() - (GetScale() / GetMaxDim()) * delta;
      SetOffset(new_offset);
      last_pos_ = loc;
    } else {
      Gtk::DrawingArea::on_motion_notify_event(event);
      GdkEventMotion transformed = *event;
      TransformEvent(&transformed);
      sig_motion.emit(&transformed);
    }

    return true;
  }

  virtual bool on_button_press_event(GdkEventButton* event) {
    if (active_ && (event->button == 3)) {
      last_pos_ = RawPoint(event->x, event->y);
    } else {
      Gtk::DrawingArea::on_button_press_event(event);
      GdkEventButton transformed = *event;
      TransformEvent(&transformed);
      sig_button.emit(&transformed);
    }
    return true;
  }

  virtual bool on_scroll_event(GdkEventScroll* event) {
    if (active_) {
      // after the change in scale, we want the mouse pointer to be over
      // the same location in the scaled view
      Eigen::Vector2d raw_point = RawPoint(event->x, event->y);
      Eigen::Vector2d center_point = TransformPoint(event->x, event->y);

      if (event->direction == GDK_SCROLL_UP) {
        if (scale_) {
          scale_->set_value(GetScale() * GetScaleRate());
        }
      } else if (event->direction == GDK_SCROLL_DOWN) {
        if (scale_) {
          scale_->set_value(GetScale() / GetScaleRate());
        }
      } else {
        Gtk::DrawingArea::on_scroll_event(event);
        return true;
      }

      // center_point = offset + raw_point * scale / max_dim
      // offset = center_point - raw_point * scale / max_dim
      SetOffset(center_point - raw_point * (GetScale() / GetMaxDim()));
    }
    return true;
  }

  virtual bool on_draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
    // draw a white rectangle for the background
    ctx->rectangle(0, 0, get_allocated_width(), get_allocated_height());
    ctx->set_source_rgb(1, 1, 1);
    ctx->fill_preserve();
    ctx->set_source_rgb(0, 0, 0);
    ctx->stroke();

    // scale and translate so that we can draw in cartesian coordinates
    ctx->save();
    // make it so that drawing commands in the virtual space map to pixel
    // coordinates
    ctx->scale(GetMaxDim() / GetScale(), -GetMaxDim() / GetScale());
    ctx->translate(0, -GetScale() * get_allocated_height() / GetMaxDim());
    ctx->translate(-GetOffset()[0], -GetOffset()[1]);

    ctx->set_line_width(0.001);

    sig_draw.emit(ctx);
    ctx->restore();

    return true;
  }
};

}  // curves
}  // gtk

#endif  // MPBLOCKS_GTK_PAN_ZOOM_VIEW_H_
