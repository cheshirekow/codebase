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
#include <mpblocks/gtk.hpp>
#include <mpblocks/util/path_util.h>


using namespace mpblocks;


// Acts just like a T but has a distinct type from any other T
template <typename T, int i>
struct UniqueType {
  T storage;
  UniqueType() : storage(0) {}
  UniqueType(T value) : storage(value) {}
  operator T&() { return storage; }
  operator const T&() const { return storage; }
};

struct Traits {
  typedef Traits This;
  static const int NDim = 2;

  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, NDim, 1> Point;
  typedef edelsbrunner96::SimplexBase<This> Simplex;

  typedef std::vector<Point> PointList;
  typedef std::vector<Simplex> SimplexList;

  typedef UniqueType<int,0> PointRef;
  typedef UniqueType<int,1> SimplexRef;

  class Storage {
   public:
    SimplexList simplices;
    PointList points;
    std::set<int> free;

    ~Storage() { Clear(); }

    const Point& operator[](const PointRef p) { return points[p]; }

    PointRef NullPoint() { return -1; }

    Simplex& operator[](SimplexRef s) { return simplices[s]; }

    SimplexRef NullSimplex() { return -1; }

    SimplexRef Promote() {
      SimplexRef result = -1;
      if (!free.empty()) {
        auto iter = free.begin();
        result = *iter;
        free.erase(iter);
      } else {
        simplices.emplace_back();
        simplices.back().version = 1;
        result = simplices.size()-1;
      }
      return result;
    }

    void Retire(SimplexRef s) {
      simplices[s].version++;
      free.insert(s);
    }

    void Clear() {
      simplices.clear();
      points.clear();
      free.clear();
    }

    PointRef AddPoint(const Point& p) {
      points.push_back(p);
      return points.size() - 1;
    }
  };
};

class Main {
 private:
  Gtk::Main gtkmm_;  //< note: must be first
  gtk::LayoutMap layout_;
  gtk::PanZoomView view_;
  sigc::connection point_tool_cnx_;

  Traits::Storage storage_;
  Traits::SimplexRef designated_simplex_;

 public:
  Main();

  ~Main() {
    layout_.saveValues(path_util::GetPreferencesPath() +
                       "/edelsbrunner96_demo_2d.yaml");
  }

  void OnMouseMotionView(GdkEventMotion* event) {
    layout_.get<Gtk::Adjustment>("view_x")->set_value(event->x);
    layout_.get<Gtk::Adjustment>("view_y")->set_value(event->y);
  }

  void AddPoint(GdkEventButton* event) {
    if(event->button != 1) {
      return;
    }
    Eigen::Vector2d point(event->x, event->y);
    if (storage_.points.size() < 3) {
      storage_.AddPoint(point);
      if (storage_.points.size() == 3) {
        designated_simplex_ =
            edelsbrunner96::Triangulate<Traits>(storage_, {0, 1, 2});
      }
    } else {
      Traits::PointRef point_ref = storage_.AddPoint(point);
      designated_simplex_ = edelsbrunner96::FuzzyWalkInsert<Traits>(
          storage_, designated_simplex_, point_ref, 1e-9);
    }
    view_.queue_draw();
  }

  void Reset() {
    storage_.Clear();
    designated_simplex_ = storage_.NullSimplex();
    view_.queue_draw();
  }

 private:
 public:
  void Run() { gtkmm_.run(*(layout_.widget<Gtk::Window>("main"))); }
  void Draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

Main::Main() {
  std::vector<std::string> paths = {
      path_util::GetSourcePath() + "/src/edelsbrunner96/demo_2d.glade",
      path_util::GetResourcePath() + "/edelsbrunner96_demo_2d.glade"};

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

  view_.SetOffsetAdjustments(layout_.get<Gtk::Adjustment>("view_offset_x"),
                             layout_.get<Gtk::Adjustment>("view_offset_y"));
  view_.SetScaleAdjustments(layout_.get<Gtk::Adjustment>("view_scale"),
                            layout_.get<Gtk::Adjustment>("view_scale_rate"));

  Gtk::Box* view_box = layout_.get<Gtk::Box>("view_box");
  if (view_box) {
    view_box->pack_start(view_, true, true);
    view_box->reorder_child(view_, 0);
    view_.show();
  } else {
    std::cerr << "There is no GtkBox named 'view_box' in the gladefile\n";
  }

  // load default settings if available
  try {
    std::string default_file =
        path_util::GetPreferencesPath() + "/edelsbrunner96_demo_2d.yaml";
    layout_.loadValues(default_file);
  } catch (const YAML::BadFile& ex) {
    std::cerr << "No default settings available";
  }

  view_.sig_motion.connect(sigc::mem_fun(this, &Main::OnMouseMotionView));
  view_.sig_button.connect(sigc::mem_fun(this, &Main::AddPoint));
  view_.sig_draw.connect(sigc::mem_fun(this, &Main::Draw));

  // when drawing options change, redraw the scene
  for (const std::string chk_key :
       {"draw_sites", "draw_circumcircles", "draw_delaunay", "draw_voronoi",
        "draw_active_cell", "draw_vertex_labels", "draw_simplex_labels",
        "draw_neighbor_labels"}) {
    layout_.get<Gtk::CheckButton>(chk_key)->signal_toggled().connect(
        sigc::mem_fun(&view_, &Gtk::DrawingArea::queue_draw));
  }

  for (const std::string adj_key :
       {"site_radius", "site_stroke_width", "circumcircle_stroke_width",
        "delaunay_edge_width", "voronoi_edge_width"}) {
    layout_.get<Gtk::Adjustment>(adj_key)->signal_value_changed().connect(
        sigc::mem_fun(&view_, &Gtk::DrawingArea::queue_draw));
  }

  for (const std::string color_key :
       {"site_fill_color", "site_stroke_color", "circumcircle_fill_color",
        "circumcircle_stroke_color", "delaunay_edge_color",
        "voronoi_edge_color", "active_triangle_color"}) {
    layout_.get<Gtk::ColorButton>(color_key)->signal_color_set().connect(
        sigc::mem_fun(&view_, &Gtk::DrawingArea::queue_draw));
  }

  layout_.get<Gtk::Button>("reset")->signal_clicked().connect(
      sigc::mem_fun(this, &Main::Reset));

  // activate pan-zoom by default
  layout_.get<Gtk::ToggleButton>("pan_zoom")->set_active(true);
}

void Main::Draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  gtk::EigenCairo ectx(ctx);
  if (layout_.get<Gtk::CheckButton>("draw_circumcircles")->get_active()) {
    Gdk::RGBA fill_color =
        layout_.get<Gtk::ColorButton>("circumcircle_fill_color")->get_rgba();
    Gdk::RGBA stroke_color =
        layout_.get<Gtk::ColorButton>("circumcircle_stroke_color")->get_rgba();
    double scale = layout_.get<Gtk::Adjustment>("view_scale")->get_value();
    double stroke_width =
        layout_.get<Gtk::Adjustment>("circumcircle_stroke_width")->get_value() /
        10000;

    ctx->set_line_width(stroke_width * scale);
    for (std::size_t i=0; i < storage_.simplices.size(); i++) {
      if(storage_.free.count(i)) {
        continue;
      }
      auto& simplex = storage_.simplices[i];
      if (simplex.V[0] == storage_.NullPoint()) {
        continue;
      }
      simplex.ComputeCenter(storage_);
      ectx.circle(simplex.c, std::sqrt(simplex.r2));
      ectx.set_source(fill_color);
      ctx->fill_preserve();
      ectx.set_source(stroke_color);
      ctx->stroke();
    }
  }

  if (layout_.get<Gtk::CheckButton>("draw_delaunay")->get_active()) {
    Gdk::RGBA stroke_color =
        layout_.get<Gtk::ColorButton>("delaunay_edge_color")->get_rgba();
    double scale = layout_.get<Gtk::Adjustment>("view_scale")->get_value();
    double stroke_width =
        layout_.get<Gtk::Adjustment>("delaunay_edge_width")->get_value() /
        10000;

    ctx->set_line_cap(Cairo::LINE_CAP_ROUND);
    ctx->set_line_join(Cairo::LINE_JOIN_ROUND);
    ctx->set_line_width(stroke_width * scale);
    ectx.set_source(stroke_color);
    for (std::size_t i=0; i < storage_.simplices.size(); i++) {
      if(storage_.free.count(i)) {
        continue;
      }
      auto& simplex = storage_.simplices[i];
      if (simplex.V[0] == storage_.NullPoint()) {
        continue;
      }
      ectx.move_to(storage_[simplex.V[0]]);
      ectx.line_to(storage_[simplex.V[1]]);
      ectx.line_to(storage_[simplex.V[2]]);
      ectx.line_to(storage_[simplex.V[0]]);
      ctx->stroke();
    }
  }

  if (layout_.get<Gtk::CheckButton>("draw_voronoi")->get_active()) {
    Gdk::RGBA stroke_color = layout_.get<Gtk::ColorButton>("voronoi_edge_color")
        ->get_rgba();
    double scale = layout_.get<Gtk::Adjustment>("view_scale")->get_value();
    double stroke_width = layout_.get<Gtk::Adjustment>("voronoi_edge_width")
        ->get_value() / 10000;

    ctx->set_line_cap(Cairo::LINE_CAP_ROUND);
    ctx->set_line_join(Cairo::LINE_JOIN_ROUND);
    ctx->set_line_width(stroke_width * scale);
    ectx.set_source(stroke_color);

    for (std::size_t i = 0; i < storage_.simplices.size(); i++) {
      if (storage_.free.count(i)) {
        continue;
      }
      auto& simplex = storage_.simplices[i];
      if (simplex.V[0] == storage_.NullPoint()) {
        continue;
      }
      for (int j = 0; j < 3; j++) {
        size_t k = simplex.N[j];
        if (k < 0 || k >= storage_.simplices.size()) {
          continue;
        }
        auto& neighbor = storage_.simplices[k];
        if(neighbor.V[0] == storage_.NullPoint()) {
          continue;
        }
        ectx.move_to(simplex.c);
        ectx.line_to(neighbor.c);
        ctx->stroke();
      }
    }
  }

  if (layout_.get<Gtk::CheckButton>("draw_sites")->get_active()) {
    Gdk::RGBA fill_color = layout_.get<Gtk::ColorButton>("site_fill_color")
        ->get_rgba();
    Gdk::RGBA stroke_color = layout_.get<Gtk::ColorButton>("site_stroke_color")
        ->get_rgba();
    double radius = layout_.get<Gtk::Adjustment>("site_radius")->get_value()
        / 1000;
    double scale = layout_.get<Gtk::Adjustment>("view_scale")->get_value();
    double stroke_width = layout_.get<Gtk::Adjustment>("site_stroke_width")
        ->get_value() / 10000;

    ctx->set_line_width(stroke_width * scale);
    for (const Eigen::Vector2d& point : storage_.points) {
      ectx.circle(point, radius * scale);
      ectx.set_source(fill_color);
      ctx->fill_preserve();
      ectx.set_source(stroke_color);
      ctx->stroke();
    }
  }

  Glib::RefPtr<Pango::Layout> layout = Pango::Layout::create(ctx);
  Pango::FontDescription font_desc("Sans 6");
  layout->set_font_description(font_desc);
  if (layout_.get<Gtk::CheckButton>("draw_vertex_labels")->get_active()) {
    ctx->set_source_rgb(0, 0, 0);
    for (std::size_t i = 0; i < storage_.points.size(); i++) {
      ectx.move_to(storage_.points[i]);
      ctx->save();
      ctx->scale(1, -1);
      ctx->scale(0.25, 0.25);
      layout->update_from_cairo_context(ctx);
      layout->set_text(boost::str(boost::format("[%d]") % i));
      layout->show_in_cairo_context(ctx);
      ctx->restore();
    }
  }

  if (layout_.get<Gtk::CheckButton>("draw_simplex_labels")->get_active()) {
    ctx->set_source_rgb(0, 0, 0);
    for (std::size_t i = 0; i < storage_.simplices.size(); i++) {
      if (storage_.free.count(i)) {
        continue;
      }
      auto& simplex = storage_.simplices[i];
      Eigen::Vector2d center(0,0);
      if (simplex.V[0] == storage_.NullPoint()) {
        Eigen::Vector2d edge_center =
            (1 / 2.0) * (storage_[simplex.V[1]] + storage_[simplex.V[2]]);
        auto neighbor_ref = simplex.N[0];
        auto& neighbor = storage_[neighbor_ref];
        Eigen::Vector2d neighbor_center =
            (1 / 3.0) * (storage_[neighbor.V[0]] + storage_[neighbor.V[1]] +
                         storage_[neighbor.V[2]]);
        center = 2*edge_center - neighbor_center;
      } else {
        center = (1 / 3.0) * (storage_[simplex.V[0]] + storage_[simplex.V[1]] +
                              storage_[simplex.V[2]]);
      }
      ectx.move_to(center);
      ctx->save();
      ctx->scale(1, -1);
      ctx->scale(0.25, 0.25);
      layout->update_from_cairo_context(ctx);
      layout->set_text(boost::str(boost::format("(%d)") % i));
      layout->show_in_cairo_context(ctx);
      ctx->restore();
    }
  }

  if (layout_.get<Gtk::CheckButton>("draw_neighbor_labels")->get_active()) {
    ctx->set_source_rgb(0, 0, 0);
    for (std::size_t i = 0; i < storage_.simplices.size(); i++) {
      if (storage_.free.count(i)) {
        continue;
      }
      auto& simplex = storage_.simplices[i];
      if (simplex.V[0] == storage_.NullPoint()) {
        Eigen::Vector2d edge_center =
            (1 / 2.0) * (storage_[simplex.V[1]] + storage_[simplex.V[2]]);
        auto neighbor_ref = simplex.N[0];
        auto& neighbor = storage_[neighbor_ref];
        Eigen::Vector2d neighbor_center =
            (1 / 3.0) * (storage_[neighbor.V[0]] + storage_[neighbor.V[1]] +
                         storage_[neighbor.V[2]]);
        Eigen::Vector2d center = 2 * edge_center - neighbor_center;
        /* finite edge */ {
          Eigen::Vector2d normal_to_center =
              (center - edge_center).normalized();
          Eigen::Vector2d text_at = edge_center + 3 * normal_to_center;
          ectx.move_to(text_at);
          ctx->save();
          ctx->scale(1, -1);
          ctx->scale(0.25, 0.25);
          layout->update_from_cairo_context(ctx);
          layout->set_text(boost::str(boost::format("|%d|") % neighbor_ref));
          layout->show_in_cairo_context(ctx);
          ctx->restore();
        }

        // infinite edges
        for (int j = 1; j < 3; j++) {
          neighbor_ref = simplex.N[j];
          int k = 1 + (2 - j);
          Eigen::Vector2d vertex = storage_[simplex.V[k]];
          Eigen::Vector2d normal_to_center = (center-vertex).normalized();
          Eigen::Vector2d text_at = vertex + 5 * normal_to_center;
          ectx.move_to(text_at);
          ctx->save();
          ctx->scale(1, -1);
          ctx->scale(0.25, 0.25);
          layout->update_from_cairo_context(ctx);
          layout->set_text(boost::str(boost::format("|%d|") % neighbor_ref));
          layout->show_in_cairo_context(ctx);
          ctx->restore();
        }
      } else {
        Eigen::Vector2d center =
            (1 / 3.0) * (storage_[simplex.V[0]] + storage_[simplex.V[1]] +
                         storage_[simplex.V[2]]);
        for(int j=0; j < 3; j++) {
          auto neighbor_ref = simplex.N[j];
          Eigen::Vector2d edge_center(0,0);
          for(int k=0; k < 3; k++) {
            if(k != j) {
              edge_center += storage_[simplex.V[k]];
            }
          }
          edge_center = 0.5 * edge_center;
          Eigen::Vector2d normal_to_center = (center - edge_center).normalized();
          Eigen::Vector2d text_at = edge_center + 3*normal_to_center;
          ectx.move_to(text_at);
          ctx->save();
          ctx->scale(1, -1);
          ctx->scale(0.25, 0.25);
          layout->update_from_cairo_context(ctx);
          layout->set_text(boost::str(boost::format("|%d|") % neighbor_ref));
          layout->show_in_cairo_context(ctx);
          ctx->restore();
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  Main app;
  app.Run();
  return 0;
}
