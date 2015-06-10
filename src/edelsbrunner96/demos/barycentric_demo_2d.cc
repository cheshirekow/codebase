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
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <Eigen/Dense>
#include <gtkmm.h>
#include <re2/re2.h>

#include <edelsbrunner96/edelsbrunner96.hpp>
#include <ck_gtk/eigen_cairo_impl.h>
#include <ck_gtk/layout_map.h>
#include <ck_gtk/pan_zoom_view.h>
#include <mpblocks/util/path_util.h>

using namespace mpblocks;

struct Traits {
  typedef Traits This;
  static const int NDim = 2;

  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, NDim, 1> Point;
  typedef edelsbrunner96::SimplexBase<This> Simplex;

  typedef Point* PointRef;
  typedef Simplex* SimplexRef;

  class Storage {
   public:
    const Point& operator[](const PointRef p) { return *p; }
    PointRef NullPoint() { return nullptr; }
    Simplex& operator[](SimplexRef s) { return *s; }
    SimplexRef NullSimplex() { return nullptr; }
  };
};

class Main {
 private:
  Gtk::Main gtkmm_;  //< note: must be first
  ck_gtk::LayoutMap layout_;
  ck_gtk::PanZoomView view_;

  Traits::Storage storage_;
  Traits::Simplex simplex_;
  std::list<Traits::Point> points_;
  std::vector<Traits::PointRef> nearest_feature_;
  Traits::Point nearest_point_;
  Traits::Point query_point_;

 public:
  Main();

  ~Main() {
    layout_.SaveValues(path_util::GetPreferencesPath() +
                       "/edelsbrunner96_barycentric_demo_2d.yaml");
  }

  void OnMouseMotionView(GdkEventMotion* event) {
    if (event->state & GDK_BUTTON1_MASK) {
      Eigen::Vector2d point(event->x, event->y);
      layout_.Get<Gtk::Adjustment>("view_x")->set_value(point[0]);
      layout_.Get<Gtk::Adjustment>("view_y")->set_value(point[1]);
      if (points_.size() >= 3) {
        auto i = points_.begin();
        simplex_.V[0] = &(*i++);
        simplex_.V[1] = &(*i++);
        simplex_.V[2] = &(*i++);
        auto L = simplex_.BarycentricCoordinates(storage_, point);
        layout_.Get<Gtk::Adjustment>("lambda_0")->set_value(L[0]);
        layout_.Get<Gtk::Adjustment>("lambda_1")->set_value(L[1]);
        layout_.Get<Gtk::Adjustment>("lambda_2")->set_value(L[2]);
        nearest_feature_.clear();
        query_point_ = point;
        auto dist = edelsbrunner96::SimplexDistance<Traits>(
            storage_, point, &simplex_, &nearest_point_,
            std::back_inserter(nearest_feature_));
        layout_.Get<Gtk::Adjustment>("simplex_distance")->set_value(dist);
        view_.queue_draw();
      }
    }
  }

  void AddPoint(GdkEventButton* event) {
    if (event->button == 1) {
      Eigen::Vector2d point(event->x, event->y);
      if (points_.size() < 3) {
        points_.push_back(point);
        view_.queue_draw();
      } else {
        GdkEventMotion fake_event;
        fake_event.x = event->x;
        fake_event.y = event->y;
        fake_event.state = GDK_BUTTON1_MASK;
        OnMouseMotionView(&fake_event);
      }
    }
  }

  void Reset() {
    points_.clear();
    nearest_feature_.clear();
    nearest_point_.fill(0);
    view_.queue_draw();
  }

 private:
 public:
  void Run() { gtkmm_.run(*(layout_.GetWidget<Gtk::Window>("main"))); }
  void Draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

Main::Main() {
  std::vector<std::string> paths = {
      path_util::GetSourcePath() +
          "/src/edelsbrunner96/demos/barycentric_demo_2d.glade",
      path_util::GetResourcePath() +
          "/edelsbrunner96_barycentric_demo_2d.glade"};

  bool layout_loaded = false;
  for (const std::string path : paths) {
    layout_loaded = layout_.loadLayout(path);
    if (layout_loaded) {
      break;
    }
  }
  if (!layout_loaded) {
    std::cerr << "Layout not loaded!";
    exit(1);
  }

  view_.SetOffsetAdjustments(layout_.Get<Gtk::Adjustment>("view_offset_x"),
                             layout_.Get<Gtk::Adjustment>("view_offset_y"));
  view_.SetScaleAdjustments(layout_.Get<Gtk::Adjustment>("view_scale"),
                            layout_.Get<Gtk::Adjustment>("view_scale_rate"));

  Gtk::Box* view_box = layout_.Get<Gtk::Box>("view_box");
  if (view_box) {
    view_box->pack_start(view_, true, true);
    view_box->reorder_child(view_, 0);
    view_.show();
  } else {
    std::cerr << "There is no GtkBox named 'view_box' in the gladefile\n";
  }

  // load default settings if available
  try {
    std::string default_file = path_util::GetPreferencesPath() +
                               "/edelsbrunner96_barycentric_demo_2d.yaml";
    layout_.LoadValues(default_file);
  } catch (const YAML::BadFile& ex) {
    std::cerr << "No default settings available";
  }

  view_.sig_motion.connect(sigc::mem_fun(this, &Main::OnMouseMotionView));
  view_.sig_draw.connect(sigc::mem_fun(this, &Main::Draw));
  view_.sig_button.connect(sigc::mem_fun(this, &Main::AddPoint));

  // when drawing options change, redraw the scene
  for (const std::string chk_key :
       {"draw_sites", "draw_edges", "draw_nearest_feature",
        "draw_distance_segment"}) {
    layout_.Get<Gtk::CheckButton>(chk_key)->signal_toggled().connect(
        sigc::mem_fun(&view_, &Gtk::DrawingArea::queue_draw));
  }

  for (const std::string adj_key :
       {"site_radius", "site_stroke_width", "edge_stroke_width",
        "feature_radius", "distance_stroke_width", "feature_stroke_width"}) {
    layout_.Get<Gtk::Adjustment>(adj_key)->signal_value_changed().connect(
        sigc::mem_fun(&view_, &Gtk::DrawingArea::queue_draw));
  }

  for (const std::string color_key :
       {"site_fill_color", "site_stroke_color", "edge_stroke_color",
        "feature_fill_color", "feature_stroke_color",
        "distance_stroke_color"}) {
    layout_.Get<Gtk::ColorButton>(color_key)->signal_color_set().connect(
        sigc::mem_fun(&view_, &Gtk::DrawingArea::queue_draw));
  }

  layout_.Get<Gtk::Button>("reset")->signal_clicked().connect(
      sigc::mem_fun(this, &Main::Reset));
}

void Main::Draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  ck_gtk::EigenCairo ectx(ctx);
  if (layout_.Get<Gtk::CheckButton>("draw_sites")->get_active()) {
    Gdk::RGBA fill_color =
        layout_.Get<Gtk::ColorButton>("site_fill_color")->get_rgba();
    Gdk::RGBA stroke_color =
        layout_.Get<Gtk::ColorButton>("site_stroke_color")->get_rgba();
    double radius =
        layout_.Get<Gtk::Adjustment>("site_radius")->get_value() / 1000;
    double scale = layout_.Get<Gtk::Adjustment>("view_scale")->get_value();
    double stroke_width =
        layout_.Get<Gtk::Adjustment>("site_stroke_width")->get_value() / 10000;

    ctx->set_line_width(stroke_width * scale);
    for (const Eigen::Vector2d& point : points_) {
      ectx.circle(point, radius * scale);
      ectx.set_source(fill_color);
      ctx->fill_preserve();
      ectx.set_source(stroke_color);
      ctx->stroke();
    }
  }

  if (layout_.Get<Gtk::CheckButton>("draw_edges")->get_active()) {
    Gdk::RGBA stroke_color =
        layout_.Get<Gtk::ColorButton>("edge_stroke_color")->get_rgba();
    double scale = layout_.Get<Gtk::Adjustment>("view_scale")->get_value();
    double stroke_width =
        layout_.Get<Gtk::Adjustment>("edge_stroke_width")->get_value() / 10000;

    ctx->set_line_cap(Cairo::LINE_CAP_ROUND);
    ctx->set_line_join(Cairo::LINE_JOIN_ROUND);
    ctx->set_line_width(stroke_width * scale);
    ectx.set_source(stroke_color);
    ctx->set_line_width(stroke_width * scale);
    ectx.move_to(points_.back());
    for (const Eigen::Vector2d& point : points_) {
      ectx.line_to(point);
    }
    ctx->stroke();
  }

  if (layout_.Get<Gtk::CheckButton>("draw_nearest_feature")->get_active()) {
    Gdk::RGBA fill_color =
        layout_.Get<Gtk::ColorButton>("feature_fill_color")->get_rgba();
    Gdk::RGBA stroke_color =
        layout_.Get<Gtk::ColorButton>("feature_stroke_color")->get_rgba();
    double radius =
        layout_.Get<Gtk::Adjustment>("feature_radius")->get_value() / 1000;
    double scale = layout_.Get<Gtk::Adjustment>("view_scale")->get_value();
    double stroke_width =
        layout_.Get<Gtk::Adjustment>("feature_stroke_width")->get_value() /
        10000;
    ctx->set_line_width(stroke_width * scale);

    switch (nearest_feature_.size()) {
      case 1: {
        ectx.circle(storage_[nearest_feature_[0]], radius * scale);
        ectx.set_source(fill_color);
        ctx->fill_preserve();
        ectx.set_source(stroke_color);
        ctx->stroke();
        break;
      }
      case 2: {
        ectx.move_to(storage_[nearest_feature_[0]]);
        ectx.line_to(storage_[nearest_feature_[1]]);
        ectx.set_source(stroke_color);
        ctx->stroke();
        break;
      }
      case 3: {
        ectx.move_to(storage_[nearest_feature_[2]]);
        for (auto pointref : nearest_feature_) {
          ectx.line_to(storage_[pointref]);
        }
        ectx.set_source(fill_color);
        ctx->fill_preserve();
        ectx.set_source(stroke_color);
        ctx->stroke();
        break;
      }
      default:
        break;
    }
  }

  if (layout_.Get<Gtk::CheckButton>("draw_distance_segment")->get_active() &&
      nearest_feature_.size() > 0) {
    Gdk::RGBA stroke_color =
        layout_.Get<Gtk::ColorButton>("distance_stroke_color")->get_rgba();
    double scale = layout_.Get<Gtk::Adjustment>("view_scale")->get_value();
    double stroke_width =
        layout_.Get<Gtk::Adjustment>("distance_stroke_width")->get_value() /
        10000;

    ctx->set_line_cap(Cairo::LINE_CAP_ROUND);
    ctx->set_line_join(Cairo::LINE_JOIN_ROUND);
    ctx->set_line_width(stroke_width * scale);
    ectx.set_source(stroke_color);
    ctx->set_line_width(stroke_width * scale);
    ectx.move_to(query_point_);
    ectx.line_to(nearest_point_);
    ctx->stroke();
  }

  if (layout_.Get<Gtk::CheckButton>("draw_sites")->get_active() &&
      points_.size() >= 3) {
    Gdk::RGBA fill_color =
        layout_.Get<Gtk::ColorButton>("site_fill_color")->get_rgba();
    Gdk::RGBA stroke_color =
        layout_.Get<Gtk::ColorButton>("site_stroke_color")->get_rgba();
    double radius =
        layout_.Get<Gtk::Adjustment>("site_radius")->get_value() / 1000;
    double scale = layout_.Get<Gtk::Adjustment>("view_scale")->get_value();
    double stroke_width =
        layout_.Get<Gtk::Adjustment>("site_stroke_width")->get_value() / 10000;

    ctx->set_line_width(stroke_width * scale);
    ectx.circle(query_point_, radius * scale);
    ectx.set_source(fill_color);
    ctx->fill_preserve();
    ectx.set_source(stroke_color);
    ctx->stroke();
  }
}

int main(int argc, char** argv) {
  Main app;
  app.Run();
  return 0;
}
