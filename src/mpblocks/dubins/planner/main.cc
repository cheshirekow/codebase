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
#include <Eigen/Dense>
#include <gtkmm.h>
#include <re2/re2.h>

#include <mpblocks/dubins/curves_eigen.hpp>
#include <mpblocks/dubins/curves_eigen/cairo.hpp>
#include <mpblocks/gtk.hpp>
#include <mpblocks/util/path_util.h>

namespace mpblocks {
namespace dubins {
namespace curves_eigen {

// Ripped from hyper two constraint solver
struct PointSolver {
  typedef Eigen::Matrix<double, 3, 1> Vector3d;
  typedef Eigen::Matrix<double, 2, 1> Vector2d;
  typedef Path<double> Result;

  static Result solveLS(const Vector3d& q0, const Vector2d& p,
                          const double r) {
    Result out(LSL);

    // calculate the center of the circle to which q0 is tangent
    Vector2d c = leftCenter(q0, r);
    Vector2d cp = (p - c);

    // if the solution is less than r from this center, then the Left
    // solution is infeasible
    if (cp.squaredNorm() < r * r) return out;

    // otherwise we turn left until we are pointed in the right direction,
    // then we go straight the rest of the way
    // see 2013-06-07-Note-17-51_dubins_kdtree.xoj for drawings and math
    double d_cp = cp.norm();

    // this is the interior angle of the right triangle (c,t,p) where
    // c is the circle center, p is the target point, and t is a point
    // on the circle such that c->t is perpendicularo to t->p and t->p
    // is tangent to the circle
    double alpha = std::acos(r / d_cp);

    // this is the angle that the vector c->p makes with the circle
    double beta = std::atan2(cp[1], cp[0]);

    // this is the target angle for the tangent point
    double theta1 = clampRadian(beta - alpha);

    // the is our start angle
    double theta0 = leftAngleOf(q0);

    // this is the arc distance we must travel on the circle
    double arc0 = ccwArc(theta0, theta1);

    // this is the length of the segment from t->p
    double d1 = d_cp * std::sin(alpha);

    out = Vector3d(arc0, d1, 0);
    return out;
  };

  static Result solveRS(const Vector3d& q0, const Vector2d& p,
                          const double r) {
    Result out(RSR);

    // calculate the center of the circle to which q0 is tangent
    Vector2d c = rightCenter(q0, r);

    Vector2d cp = (p - c);

    // if the solution is less than r from this center, then the Left
    // solution is infeasible
    if (cp.squaredNorm() < r * r) return out;

    // otherwise we turn left until we are pointed in the right direction,
    // then we go straight the rest of the way
    // see 2013-06-07-Note-17-51_dubins_kdtree.xoj for drawings and math
    double d_cp = cp.norm();

    // this is the interior angle of the right triangle (c,t,p) where
    // c is the circle center, p is the target point, and t is a point
    // on the circle such that c->t is perpendicularo to t->p and t->p
    // is tangent to the circle
    double alpha = std::acos(r / d_cp);

    // this is the angle that the vector c->p makes with the circle
    double beta = std::atan2(cp[1], cp[0]);

    // this is the target angle for the tangent point
    double theta1 = clampRadian(beta + alpha);

    // the is our start angle
    double theta0 = rightAngleOf(q0);

    // this is the arc distance we must travel on the circle
    double arc0 = cwArc(theta0, theta1);

    // this is the length of the segment from t->p
    double d1 = d_cp * std::sin(alpha);

    out = Vector3d(arc0, d1, 0);
    return out;
  };

  static Result solve(const Vector3d& q0, const Vector2d& p,
                        const double r) {
    return bestOf(solveLS(q0, p, r), solveRS(q0, p, r), r);
  };
};

} //< namespace curves_eigen
} //< namespace dubins
} //< nampesace mpblocks

namespace mpblocks {
namespace examples {
namespace   dubins {

using namespace mpblocks::dubins;

struct Node;

struct Edge {
  Path<double> path;
  double dist;
  Node* child;
};

bool operator<(const Edge& a, const Edge& b) {
  return (a.dist == b.dist) ? (a.child < b.child) : (a.dist < b.dist);
}

struct PlanStep {
  Eigen::Vector3d start_state;
  Node* next_node;
  Path<double> path_to;
};

struct Node {
  double mst_cost;
  Node* mst_parent;
  Eigen::Vector3d state;
  std::set<Edge> out_edges;

  Node() {
    mst_cost = std::numeric_limits<double>::max();
    mst_parent = nullptr;
    state.fill(0);
  }
};

class View : public Gtk::DrawingArea {
 private:
  Eigen::Vector2d m_virtual_offset; // virtual offset
  double m_scale;

 public:
  sigc::signal<void, const Cairo::RefPtr<Cairo::Context>&> sig_draw;
  sigc::signal<void, Eigen::Vector2d> sig_mouseMotion;
  sigc::signal<void, int, Eigen::Vector2d> sig_mousePress;
  sigc::signal<void, int, Eigen::Vector2d> sig_mouseRelease;
  sigc::signal<void, Eigen::Vector2d> sig_rawMouseMotion;
  sigc::signal<void, int, Eigen::Vector2d> sig_rawMousePress;
  sigc::signal<void, int, Eigen::Vector2d> sig_rawMouseRelease;
  sigc::signal<void, GdkScrollDirection> sig_scroll;


  View() {
    add_events(Gdk::POINTER_MOTION_MASK | Gdk::BUTTON_MOTION_MASK |
               Gdk::BUTTON_PRESS_MASK | Gdk::BUTTON_RELEASE_MASK |
               Gdk::SCROLL_MASK);
    m_scale = 1.0;
    m_virtual_offset.fill(0);
  }

  void SetViewport(double scale, const Eigen::Vector2d& offset) {
    m_scale = scale;
    m_virtual_offset = offset;
    queue_draw();
  }

  Eigen::Vector2d TransformPoint(double x, double y) {
    y = get_allocated_height() - y;
    int max_dim = std::max(get_allocated_width(), get_allocated_height());
    double virtual_x = m_virtual_offset[0] +
                       x * m_scale / max_dim;
    double virtual_y = m_virtual_offset[1] +
                       y * m_scale / max_dim;
    return Eigen::Vector2d(virtual_x, virtual_y);
  }

  Eigen::Vector2d RawPoint(double x, double y) {
      y = get_allocated_height() - y;
      int max_dim = std::max(get_allocated_width(), get_allocated_height());
      double virtual_x = x / max_dim;
      double virtual_y = y / max_dim;
      return Eigen::Vector2d(virtual_x, virtual_y);
    }

  virtual bool on_motion_notify_event(GdkEventMotion* event) {
    sig_mouseMotion.emit(TransformPoint(event->x, event->y));
    sig_rawMouseMotion.emit(RawPoint(event->x, event->y));
    return true;
  }

  virtual bool on_button_press_event(GdkEventButton* event) {
    sig_mousePress.emit(event->button, TransformPoint(event->x, event->y));
    sig_rawMousePress.emit(event->button, RawPoint(event->x, event->y));
    return true;
  }

  virtual bool on_button_release_event(GdkEventButton* event) {
    sig_mouseRelease.emit(event->button, TransformPoint(event->x, event->y));
    sig_rawMouseRelease.emit(event->button, RawPoint(event->x, event->y));
    return true;
  }

  virtual void on_size_allocate(Gtk::Allocation& allocation) {
    Gtk::DrawingArea::on_size_allocate(allocation);
  }

  virtual bool on_scroll_event(GdkEventScroll* event) {
    sig_scroll.emit(event->direction);
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
    int max_dim = std::max(get_allocated_width(), get_allocated_height());
    ctx->scale(max_dim / m_scale,
               -max_dim / m_scale);
    ctx->translate(0, -m_scale * get_allocated_height() / max_dim);
    ctx->translate(-m_virtual_offset[0], -m_virtual_offset[1]);

    ctx->set_line_width(0.001);

    sig_draw.emit(ctx);
    ctx->restore();

    return true;
  }
};

class ObstacleTool {
 private:
  std::list<Eigen::Vector2d>  m_points;
  Eigen::Vector2d m_next;

 public:
  sigc::signal<void,std::list<Eigen::Vector2d>*> sig_finished;
  sigc::signal<void> sig_changed;

  void Cancel() {
    m_points.clear();
  }

  void Activate(View* view, std::list<sigc::connection>* tool_cnx) {
    auto cnx = std::back_inserter(*tool_cnx);
    *cnx++ = view->sig_mousePress.connect(
        sigc::mem_fun(this, &ObstacleTool::OnMousePress));
    *cnx++ = view->sig_mouseMotion.connect(
            sigc::mem_fun(this, &ObstacleTool::OnMouseMotion));
    *cnx++ = view->sig_draw.connect(
            sigc::mem_fun(this, &ObstacleTool::Draw));
    *cnx++ = sig_changed.connect(
           sigc::mem_fun(view, &Gtk::DrawingArea::queue_draw));
  }

  void OnMousePress(int button, Eigen::Vector2d loc) {
    if(button == 1) {
      m_points.push_back(loc);
      m_next = loc;
      sig_changed.emit();
    } else {
      sig_finished.emit(&m_points);
    }
  }

  void OnMouseMotion(Eigen::Vector2d loc) {
    m_next = loc;
    sig_changed.emit();
  }

  void Draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
    if(m_points.size() < 1) {
      return;
    }

    mpblocks::gtk::EigenCairo ectx(ctx);
    std::list<Eigen::Vector2d>::iterator iter = m_points.begin();
    ectx.move_to(*iter++);
    for(; iter != m_points.end(); ++iter) {
      ectx.line_to(*iter);
    }
    ctx->set_source_rgb(1,0,0);
    ctx->fill_preserve();
    ctx->set_source_rgb(0,0,0);
    ctx->stroke();
  }
};

class PanZoomTool {
 private:
  double m_scale;
  double m_zoom_rate;
  Eigen::Vector2d m_offset;
  Eigen::Vector2d m_start_pan;
  bool m_panning;

 public:
  sigc::signal<void,double,Eigen::Vector2d> sig_viewport;

  PanZoomTool() {
    m_scale = 1.0;
    m_zoom_rate = 1.5;
    m_offset.fill(0);
    m_panning = false;
  }

  void SetZoomRate(double zoom_rate) {
    m_zoom_rate = zoom_rate;
  }

  void SetViewport(double scale, Eigen::Vector2d offset) {
    m_offset = offset;
    m_scale = scale;
  }

  void Activate(View* view, std::list<sigc::connection>* tool_cnx) {
    auto cnx = std::back_inserter(*tool_cnx);
    *cnx++ = view->sig_rawMousePress.connect(
        sigc::mem_fun(this, &PanZoomTool::OnMousePress));
    *cnx++ = view->sig_rawMouseRelease.connect(
        sigc::mem_fun(this, &PanZoomTool::OnMouseRelease));
    *cnx++ = view->sig_rawMouseMotion.connect(
        sigc::mem_fun(this, &PanZoomTool::OnMouseMotion));
    *cnx++ = view->sig_scroll.connect(
            sigc::mem_fun(this, &PanZoomTool::OnScroll));
    *cnx++ = sig_viewport.connect(sigc::mem_fun(view, &View::SetViewport));
    m_panning = false;
  }

  void OnScroll(GdkScrollDirection dir) {
    switch(dir) {
      case GDK_SCROLL_UP:
        m_scale *= m_zoom_rate;
        break;
      case GDK_SCROLL_DOWN:
        m_scale /= m_zoom_rate;
        break;
      default:
        break;
    }
    sig_viewport.emit(m_scale,m_offset);
  }

  void OnMousePress(int button, Eigen::Vector2d loc) {
    if (button == 1) {
      std::cout << "start pan\n";
      m_start_pan = loc;
      m_panning = true;
    } else {
      std::cout << "Button press: " << button << "\n";
    }
  }

  void OnMouseRelease(int button, Eigen::Vector2d loc) {
    if (button == 1) {
      std::cout << "end pan\n";
      m_offset = m_offset - m_scale*(loc - m_start_pan);
      m_panning = false;
      sig_viewport.emit(m_scale,m_offset);
    }
  }

  void OnMouseMotion(Eigen::Vector2d loc) {
    if(m_panning) {
      Eigen::Vector2d offset = m_scale*(loc - m_start_pan);
      std::cout << "Offset: " << offset.transpose() << "\n";
      sig_viewport.emit(m_scale,m_offset - offset);
    }
  }
};

class StatePickTool {
 public:
  sigc::signal<void,Eigen::Vector2d> sig_picked;
  sigc::signal<void,Eigen::Vector2d> sig_removed;

  void Activate(View* view, std::list<sigc::connection>* tool_cnx) {
    auto cnx = std::back_inserter(*tool_cnx);
    *cnx++ = view->sig_mousePress.connect(
        sigc::mem_fun(this, &StatePickTool::OnMousePress));
  }

  void OnMousePress(int button, Eigen::Vector2d loc) {
    if(button == 1) {
      sig_picked.emit(loc);
    } else {
      sig_removed.emit(loc);
    }
  }
};



class Main {
 private:
  Gtk::Main m_gtkmm;  //< note: must be first
  gtk::LayoutMap m_layout;
  View  m_view;
  std::set<std::string> m_settings;
  bool m_protect_restore_settings;
  bool m_protect_viewport;
  std::list<sigc::connection> m_tool_cnx;
  ObstacleTool m_obstacle_tool;
  PanZoomTool m_pan_zoom;
  StatePickTool m_state_pick_tool;
  std::list<std::list<Eigen::Vector2d>> m_obstacles;
  std::list<PlanStep> m_locked_plan;
  double m_anim_state;

  std::vector<Node> m_graph;
  Node m_start_node;
  Node m_goal_node;
  std::list<Node> m_ditch_graph;

 public:
  Main();

  ~Main() {
    m_layout.saveValues(path_util::GetPreferencesPath() +
                        "/dubins_planner.yaml");
  }

 private:
  void RestoreSettings();
  void SaveSettings();
  void DeleteSettings();

  void OnZoomRateChanged();
  void OnViewportChanged();
  void OnStartOrGoalChanged();

  void SetStartPoint(Eigen::Vector2d point);
  void SetGoalPoint(Eigen::Vector2d point);
  void AddDitchPoint(Eigen::Vector2d point);
  void RemoveDitchPoint(Eigen::Vector2d point);

  void OnVehicleChanged();

  void DisconnectTool();
  void OnToggleDrawObstacle();
  void OnObstacleFinished(std::list<Eigen::Vector2d>* obstacle);
  void OnTogglePanZoom();
  void OnPanZoomChanged(double scale, Eigen::Vector2d offset);

  void OnPickStartState();
  void OnPickGoalState();
  void OnPickDitchPoints();

  void Draw(const Cairo::RefPtr<Cairo::Context>& ctx);

  void LoadObstacles(const YAML::Node& node);
  void SaveObstacles(YAML::Emitter* yaml_out);
  void LoadDitchPoints(const YAML::Node& node);
  void SaveDitchPoints(YAML::Emitter* yaml_out);

  void GenerateStartEdges();
  void GenerateGraph();
  void BuildMinSpanningTree();

  bool IsCollision(Eigen::Vector2d point);
  bool IsCollision(Eigen::Vector3d start_state, Path<double> path);

  void StepAnimation();
  void OnToggleAnimation();
  bool OnAnimationTick();

  void OnLockPlan();
  void OnSaveImage();
  void SaveImage(const std::string& filename);

 public:
  void run() {
    m_gtkmm.run(*(m_layout.widget<Gtk::Window>("main")));
  }
};



Main::Main() {
  m_protect_restore_settings = false;
  m_protect_viewport = false;

  m_start_node.state.fill(0);
  m_goal_node.state.fill(0);

  std::vector<std::string> paths = {
      path_util::GetSourcePath() + "/src/dubins/planner/layout.glade",
      path_util::GetResourcePath() + "/dubins_planner_demo.glade"};

  bool layout_loaded = false;
  for (const std::string path : paths) {
    layout_loaded = m_layout.loadLayout(path);
    if (layout_loaded) {
      break;
    }
  }
  if (!layout_loaded) {
    std::cerr << "Layout not loaded!";
    exit(1);
  }

  Gtk::Box* view_box = m_layout.get<Gtk::Box>("view_box");
  if (view_box) {
    view_box->pack_start(m_view, true, true);
    view_box->reorder_child(m_view, 0);
    m_view.show();
  } else {
    std::cerr << "There is no GtkBox named 'view_box' in the gladefile\n";
  }

  // retrieve all saved settings
  Gtk::ComboBoxText* settings_combo =
             m_layout.get<Gtk::ComboBoxText>("settings_combo");
  std::cout << "Available settings\n";
  std::string settings_dir =
      path_util::GetPreferencesPath() + "/dubins_planner/";
  for (boost::filesystem::directory_iterator itr(settings_dir);
       itr != boost::filesystem::directory_iterator(); ++itr) {
    std::cout << itr->path() << "\n";
    try {
      YAML::Node doc = YAML::LoadFile(itr->path().c_str());
      if(doc["settings_name"]) {
        std::cout << "  name: " << doc["settings_name"].as<std::string>()
                  << "\n";
        m_settings.insert(doc["settings_name"].as<std::string>());
      }
    } catch (const YAML::BadFile& ex) {
      std::cerr << "Warning: " << ex.what() << std::endl;
    }
  }

  settings_combo->remove_all();
  for (const std::string& name : m_settings) {
    std::cout << name << "\n";
    settings_combo->append(name);
  }

  // connect signals (these are things whose state is stored so we want to
  // connect them before loading
  for (const std::string field : {"x", "y", "scale"}) {
    std::string name = std::string("view_") + field;
    m_layout.get<Gtk::Adjustment>(name)->signal_value_changed().connect(
        sigc::mem_fun(this, &Main::OnViewportChanged));
  }
  m_layout.get<Gtk::Adjustment>("view_zoom_rate")
      ->signal_value_changed()
      .connect(sigc::mem_fun(this, &Main::OnZoomRateChanged));

  for (const std::string& prefix : {"start_", "goal_"}) {
    for (const std::string& suffix : {"x", "y", "theta"}) {
      std::string adj_key = prefix + suffix;
      m_layout.get<Gtk::Adjustment>(adj_key)->signal_value_changed().connect(
          sigc::mem_fun(this, &Main::OnStartOrGoalChanged));
    }
  }

  for (const std::string& suffix : {"speed", "radius"}) {
    std::string adj_key = std::string("vehicle_") + suffix;
    m_layout.object<Gtk::Adjustment>(adj_key)->signal_value_changed().connect(
        sigc::mem_fun(this, &Main::OnVehicleChanged));
  }

  // load default settings if available
  try {
    std::string default_file =
        path_util::GetPreferencesPath() + "/dubins_planner.yaml";
    m_layout.loadValues(default_file);
  } catch (const YAML::BadFile& ex) {
    std::cerr << "No default settings available";
  }

  m_layout.get<Gtk::Button>("save_settings")
      ->signal_clicked()
      .connect(sigc::mem_fun(this, &Main::SaveSettings));

  m_layout.get<Gtk::Button>("delete_settings")
        ->signal_clicked()
        .connect(sigc::mem_fun(this, &Main::DeleteSettings));

  m_layout.get<Gtk::ToggleButton>("pick_obstacle")
      ->signal_toggled()
      .connect(sigc::mem_fun(this, &Main::OnToggleDrawObstacle));

  m_obstacle_tool.sig_finished.connect(
      sigc::mem_fun(this, &Main::OnObstacleFinished));

  m_view.sig_draw.connect(sigc::mem_fun(this, &Main::Draw));

  settings_combo->signal_changed().connect(
        sigc::mem_fun(this, &Main::RestoreSettings));

  m_layout.get<Gtk::ToggleButton>("pan_zoom")
        ->signal_toggled()
        .connect(sigc::mem_fun(this, &Main::OnTogglePanZoom));

  m_pan_zoom.sig_viewport.connect(sigc::mem_fun(this, &Main::OnPanZoomChanged));

  m_layout.get<Gtk::Adjustment>("draw_edge_thickness")
      ->signal_value_changed()
      .connect(sigc::mem_fun(&m_view, &Gtk::DrawingArea::queue_draw));

  m_layout.get<Gtk::Adjustment>("draw_node_radius")
      ->signal_value_changed()
      .connect(sigc::mem_fun(&m_view, &Gtk::DrawingArea::queue_draw));
  m_layout.get<Gtk::ToggleButton>("pick_start_state")
      ->signal_toggled()
      .connect(sigc::mem_fun(this, &Main::OnPickStartState));
  m_layout.get<Gtk::ToggleButton>("pick_goal_state")
      ->signal_toggled()
      .connect(sigc::mem_fun(this, &Main::OnPickGoalState));
  m_layout.get<Gtk::ToggleButton>("pick_ditch_points")
      ->signal_toggled()
      .connect(sigc::mem_fun(this, &Main::OnPickDitchPoints));
  m_layout.get<Gtk::Button>("generate_graph")
      ->signal_clicked()
      .connect(sigc::mem_fun(this, &Main::GenerateGraph));
  m_layout.get<Gtk::Button>("build_mst")->signal_clicked().connect(
      sigc::mem_fun(this, &Main::BuildMinSpanningTree));
  m_layout.get<Gtk::Button>("step_animation")->signal_clicked().connect(
      sigc::mem_fun(this, &Main::StepAnimation));
  m_layout.get<Gtk::ToggleButton>("run_animation")->signal_clicked().connect(
      sigc::mem_fun(this, &Main::OnToggleAnimation));
  m_layout.get<Gtk::Button>("lock_plan")->signal_clicked().connect(
      sigc::mem_fun(this, &Main::OnLockPlan));
  m_layout.get<Gtk::Button>("save_image")->signal_clicked().connect(
        sigc::mem_fun(this, &Main::OnSaveImage));


  // drawing check buttons
  for (const std::string& key :
       {"obstacles", "start_state", "goal_state", "graph_nodes", "ditch_points",
        "graph_edges", "initial_edges", "contingency_plan", "nominal_plan",
        "mst"}) {
    std::string chk_btn_key = std::string("draw_") + key;
    m_layout.get<Gtk::CheckButton>(chk_btn_key)
        ->signal_clicked()
        .connect(sigc::mem_fun(&m_view, &Gtk::DrawingArea::queue_draw));
    std::string color_btn_key = key + std::string("_color");
    m_layout.get<Gtk::ColorButton>(color_btn_key)
            ->signal_color_set()
            .connect(sigc::mem_fun(&m_view, &Gtk::DrawingArea::queue_draw));
  }
}

void Main::SaveSettings() {
  std::string settings_name =
      m_layout.get<Gtk::EntryBuffer>("settings_name")->get_text();

  // convert settings to yaml
  YAML::Emitter yaml_out;
  yaml_out << YAML::BeginMap;
  yaml_out << YAML::Key << "settings_name"
           << YAML::Value << settings_name
           << YAML::Key << "obstacles"
           << YAML::Value;
  SaveObstacles(&yaml_out);
  yaml_out << YAML::Key << "ditch_points"
           << YAML::Value;
  SaveDitchPoints(&yaml_out);

  // write the settings to the file
  std::string file_name = settings_name;
  RE2::GlobalReplace(&file_name, "\\W+", "_");
  std::string full_path = path_util::GetPreferencesPath() + "/dubins_planner/" +
                          file_name + ".yaml";
  try {
    boost::filesystem::create_directory(path_util::GetPreferencesPath() +
                                        "/dubins_planner/");
    std::ofstream fout(full_path.c_str());
    if (fout.good()) {
      std::cout << "Saving settings to " << full_path << "\n";
      fout << yaml_out.c_str();

      // if settings already exists in the combo box don't add it again
      if (!m_settings.count(settings_name)) {
        m_settings.insert(settings_name);
      }
    } else {
      std::cout << "Failed to save settings to " << settings_name << "\n";
    }
  } catch (...) {
    std::cout << "Failed to write to " << full_path << "\n";
  }

  m_protect_restore_settings = true;
  Gtk::ComboBoxText* settings_combo =
      m_layout.get<Gtk::ComboBoxText>("settings_combo");
  settings_combo->remove_all();
  for (const std::string& name : m_settings) {
    settings_combo->append(name);
  }
  settings_combo->set_active_text(settings_name);
  m_protect_restore_settings = false;
}

void Main::RestoreSettings() {
  // prevent recursion because loading the settings will change the entry
  if(m_protect_restore_settings) {
    return;
  }
  m_protect_restore_settings = true;
  Gtk::ComboBoxText* settings_combo =
                m_layout.get<Gtk::ComboBoxText>("settings_combo");
  std::string settings_name = settings_combo->get_active_text();
  m_layout.get<Gtk::EntryBuffer>("settings_name")->set_text(settings_name);

  std::string file_name = settings_name;
  RE2::GlobalReplace(&file_name, "\\W+", "_");
  std::string full_path = path_util::GetPreferencesPath() + "/dubins_planner/" +
                          file_name + ".yaml";
  std::cout << "Loading settings from " << full_path << "\n";
  try {
      YAML::Node doc = YAML::LoadFile(full_path);
      if(doc["obstacles"]) {
        LoadObstacles(doc["obstacles"]);
      }
      if(doc["ditch_points"]){
        LoadDitchPoints(doc["ditch_points"]);
      }
  } catch (const YAML::BadFile& ex) {
    std::cerr << "Warning: " << ex.what() << std::endl;
  }
  m_protect_restore_settings = false;
  m_view.queue_draw();
}

void Main::DeleteSettings() {
  std::cout << "delete pressed\n";
  Gtk::ComboBoxText* settings_combo =
      m_layout.get<Gtk::ComboBoxText>("settings_combo");

  std::string settings_name = settings_combo->get_active_text();
  if(!m_settings.count(settings_name)){
    std::cout << "can't find [" << settings_name << "] in settings \n";
    return;
  }
  m_settings.erase(settings_name);

  std::string file_name = settings_name;
  RE2::GlobalReplace(&file_name, "\\W+", "_");
  std::string full_path = path_util::GetPreferencesPath() + "/dubins_planner/" +
                          file_name + ".yaml";
  boost::filesystem::remove(full_path);

  m_protect_restore_settings = true;
  settings_combo->remove_all();
  for (const std::string& name : m_settings) {
    settings_combo->append(name);
  }
  m_protect_restore_settings = false;
}

void Main::OnZoomRateChanged() {
  m_pan_zoom.SetZoomRate(
      m_layout.get<Gtk::Adjustment>("view_zoom_rate")->get_value());
}

void Main::OnViewportChanged() {
  if(m_protect_viewport) {
    return;
  }
  double scale =
      m_layout.get<Gtk::Adjustment>("view_scale")->get_value();
  Eigen::Vector2d offset(
      m_layout.get<Gtk::Adjustment>("view_x")->get_value(),
      m_layout.get<Gtk::Adjustment>("view_y")->get_value());
  m_pan_zoom.SetViewport(scale, offset);
  m_view.SetViewport(scale,offset);
}

void Main::OnStartOrGoalChanged() {
  m_start_node.state = Eigen::Vector3d(
      m_layout.get<Gtk::Adjustment>("start_x")->get_value(),
      m_layout.get<Gtk::Adjustment>("start_y")->get_value(),
      m_layout.get<Gtk::Adjustment>("start_theta")->get_value() * M_PI / 180);
  m_goal_node.state = Eigen::Vector3d(
      m_layout.get<Gtk::Adjustment>("goal_x")->get_value(),
      m_layout.get<Gtk::Adjustment>("goal_y")->get_value(),
      m_layout.get<Gtk::Adjustment>("goal_theta")->get_value() * M_PI / 180);
  GenerateStartEdges();
  BuildMinSpanningTree();
  m_view.queue_draw();
}

void Main::OnLockPlan() {
  m_locked_plan.clear();
  m_anim_state = 0;
  Node* child = &m_goal_node;
  Node* parent = m_goal_node.mst_parent;
  if(!parent) {
    std::cerr << "Build the MST before locking a plan\n";
    return;
  }
  while(child != &m_start_node) {
    PlanStep step;
    step.start_state = parent->state;
    step.next_node = child;
    for(const Edge& edge: parent->out_edges) {
      if(edge.child == child) {
        step.path_to = edge.path;
        break;
      }
    }
    m_locked_plan.push_front(step);
    child = parent;
    parent = child->mst_parent;
  }
  m_view.queue_draw();
}

void Main::SetStartPoint(Eigen::Vector2d point) {
  m_layout.get<Gtk::Adjustment>("start_x")->set_value(point[0]);
  m_layout.get<Gtk::Adjustment>("start_y")->set_value(point[1]);
}

void Main::SetGoalPoint(Eigen::Vector2d point) {
  m_layout.get<Gtk::Adjustment>("goal_x")->set_value(point[0]);
  m_layout.get<Gtk::Adjustment>("goal_y")->set_value(point[1]);
}

void Main::AddDitchPoint(Eigen::Vector2d point) {
  Node node;
  node.state.fill(0);
  node.state.head<2>() = point;
  m_ditch_graph.push_back(node);
  m_view.queue_draw();
}

void Main::RemoveDitchPoint(Eigen::Vector2d point) {
  if (m_ditch_graph.size() < 1) {
    return;
  }
  Eigen::Vector3d query(0,0,0);
  query.head<2>() = point;
  auto iter = m_ditch_graph.begin();
  auto nearest_iter = iter;
  for (; iter != m_ditch_graph.end(); ++iter) {
    if ((iter->state - query).squaredNorm() <
        (nearest_iter->state - query).squaredNorm()) {
      nearest_iter = iter;
    }
  }
  m_ditch_graph.erase(nearest_iter);
  m_view.queue_draw();
}

void Main::OnVehicleChanged() {
  double speed = m_layout.object<Gtk::Adjustment>("vehicle_speed")->get_value();
  double radius =
      m_layout.object<Gtk::Adjustment>("vehicle_radius")->get_value();

  double turnRate = speed / radius;

  m_layout.object<Gtk::Adjustment>("vehicle_turn_rate")
      ->set_value(180 * turnRate / M_PI);
}

void Main::DisconnectTool() {
  for (auto cnx : m_tool_cnx) {
    cnx.disconnect();
  }
  m_tool_cnx.clear();
}

void Main::OnToggleDrawObstacle() {
  DisconnectTool();
  m_obstacle_tool.Cancel();
  if (m_layout.get<Gtk::ToggleButton>("pick_obstacle")->get_active()) {
    m_obstacle_tool.Activate(&m_view, &m_tool_cnx);
  }
  m_view.queue_draw();
}

void Main::OnObstacleFinished(std::list<Eigen::Vector2d>* obstacle) {
  if(obstacle->size() > 2) {
    std::cout << "Adding obstacle of size: " << obstacle->size() << "\n";
    m_obstacles.emplace_back();
    m_obstacles.back().swap(*obstacle);
  }
  m_layout.get<Gtk::ToggleButton>("pick_obstacle")->set_active(false);
}

void Main::OnTogglePanZoom() {
  DisconnectTool();
  OnViewportChanged();
  if (m_layout.get<Gtk::ToggleButton>("pan_zoom")->get_active()) {
    m_pan_zoom.Activate(&m_view, &m_tool_cnx);
  }
}

void Main::OnPanZoomChanged(double scale, Eigen::Vector2d offset) {
  m_protect_viewport = true;
  m_layout.get<Gtk::Adjustment>("view_scale")->set_value(scale);
  m_layout.get<Gtk::Adjustment>("view_x")->set_value(offset[0]);
  m_layout.get<Gtk::Adjustment>("view_y")->set_value(offset[1]);
  m_protect_viewport = false;
}

void Main::OnPickStartState() {
  DisconnectTool();
  if (m_layout.get<Gtk::ToggleButton>("pick_start_state")->get_active()) {
    m_state_pick_tool.Activate(&m_view, &m_tool_cnx);
    auto cnx = std::back_inserter(m_tool_cnx);
    *cnx++ = m_state_pick_tool.sig_picked.connect(
        sigc::mem_fun(this, &Main::SetStartPoint));
  }
}

void Main::OnPickGoalState() {
  DisconnectTool();
  if (m_layout.get<Gtk::ToggleButton>("pick_goal_state")->get_active()) {
    m_state_pick_tool.Activate(&m_view, &m_tool_cnx);
    auto cnx = std::back_inserter(m_tool_cnx);
    *cnx++ = m_state_pick_tool.sig_picked.connect(
        sigc::mem_fun(this, &Main::SetGoalPoint));
  }
}

void Main::OnPickDitchPoints() {
  DisconnectTool();
  if (m_layout.get<Gtk::ToggleButton>("pick_ditch_points")->get_active()) {
    m_state_pick_tool.Activate(&m_view, &m_tool_cnx);
    auto cnx = std::back_inserter(m_tool_cnx);
    *cnx++ = m_state_pick_tool.sig_picked.connect(
        sigc::mem_fun(this, &Main::AddDitchPoint));
    *cnx++ = m_state_pick_tool.sig_removed.connect(
            sigc::mem_fun(this, &Main::RemoveDitchPoint));
  }
}

void Main::OnToggleAnimation() {
  DisconnectTool();
  if (m_layout.get<Gtk::ToggleButton>("run_animation")->get_active()) {
    auto cnx = std::back_inserter(m_tool_cnx);
    *cnx++ = Glib::signal_timeout().connect(
        sigc::mem_fun(this, &Main::OnAnimationTick),
        m_layout.get<Gtk::Adjustment>("sim_period")->get_value() * 1000);
  }
}

bool Main::OnAnimationTick() {
  StepAnimation();
  return true;
}

void Main::Draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  mpblocks::gtk::EigenCairo ectx(ctx);

  ctx->set_line_width(
      m_layout.get<Gtk::Adjustment>("draw_edge_thickness")->get_value() /
      1000);

  // draw the workspace boundaries
  ctx->rectangle(
      m_layout.get<Gtk::Adjustment>("workspace_x")->get_value(),
      m_layout.get<Gtk::Adjustment>("workspace_y")->get_value(),
      m_layout.get<Gtk::Adjustment>("workspace_width")->get_value(),
      m_layout.get<Gtk::Adjustment>("workspace_height")->get_value());
  ctx->stroke();

  // draw start and goal state
  double node_radius =
      m_layout.get<Gtk::Adjustment>("draw_node_radius")->get_value() / 100;

  if(m_layout.get<Gtk::CheckButton>("draw_start_state")->get_active()) {
    ectx.set_source(
        m_layout.get<Gtk::ColorButton>("start_state_color")->get_rgba());
    ectx.circle(m_start_node.state.head<2>(), node_radius);
    ctx->fill();

    ectx.move_to(m_start_node.state.head<2>());
    Eigen::Vector2d edge_point =
        m_start_node.state.head<2>() +
        Eigen::Vector2d(node_radius * std::cos(m_start_node.state[2]),
                        node_radius * std::sin(m_start_node.state[2]));
    ectx.line_to(edge_point);
    ctx->set_source_rgb(0,0,0);
    ctx->stroke();
  }

  if (m_layout.get<Gtk::CheckButton>("draw_goal_state")->get_active()) {
    ectx.set_source(
        m_layout.get<Gtk::ColorButton>("goal_state_color")->get_rgba());
    ectx.circle(m_goal_node.state.head<2>(), node_radius);
    ctx->fill();
  }

  // draw obstacles
  if (m_layout.get<Gtk::CheckButton>("draw_obstacles")->get_active()) {
    ectx.set_source(
        m_layout.get<Gtk::ColorButton>("obstacles_color")->get_rgba());
    for (auto& list : m_obstacles) {
      std::list<Eigen::Vector2d>::iterator iter = list.begin();
      ectx.move_to(*iter++);
      for (; iter != list.end(); ++iter) {
        ectx.line_to(*iter);
      }
    }
    ctx->fill_preserve();
  }

  // draw ditch points
  if (m_layout.get<Gtk::CheckButton>("draw_ditch_points")->get_active()) {
    ectx.set_source(
        m_layout.get<Gtk::ColorButton>("ditch_points_color")->get_rgba());
    for (const Node& node : m_ditch_graph) {
      ectx.circle(node.state, node_radius);
      ctx->fill();
    }
  }

  // draw graph nodes
  if (m_layout.get<Gtk::CheckButton>("draw_graph_nodes")->get_active()) {
    for (Node& node : m_graph) {
      ectx.set_source(
          m_layout.get<Gtk::ColorButton>("graph_nodes_color")->get_rgba());
      ectx.circle(node.state.head<2>(), node_radius);
      ctx->fill();

      ectx.move_to(node.state.head<2>());
      Eigen::Vector2d edge_point =
          node.state.head<2>() +
          Eigen::Vector2d(node_radius * std::cos(node.state[2]),
                          node_radius * std::sin(node.state[2]));
      ectx.line_to(edge_point);
      ctx->set_source_rgb(0, 0, 0);
      ctx->stroke();
    }
  }

  // draw graph connections
  double radius = m_layout.get<Gtk::Adjustment>("vehicle_radius")->get_value();
  if (radius <= 0) {
    radius = 1;
  }

  curves_eigen::DrawOpts draw_opts;
  draw_opts.ctx = ctx;
  draw_opts.drawBalls = false;
  if (m_layout.get<Gtk::CheckButton>("draw_graph_edges")->get_active()) {
    draw_opts.patPath = gtk::EigenCairo::SolidPattern::create_rgba(
        m_layout.get<Gtk::ColorButton>("graph_edges_color")->get_rgba());
    for (Node& parent : m_graph) {
      for (const Edge& edge : parent.out_edges) {
        curves_eigen::GenericArtist<double>::draw(parent.state, radius,
                                                  edge.path, draw_opts);
      }
    }
  }

  if (m_layout.get<Gtk::CheckButton>("draw_initial_edges")->get_active()) {
    draw_opts.patPath = gtk::EigenCairo::SolidPattern::create_rgba(
        m_layout.get<Gtk::ColorButton>("initial_edges_color")->get_rgba());
    for (const Edge& edge : m_start_node.out_edges) {
      curves_eigen::GenericArtist<double>::draw(m_start_node.state, radius,
                                                edge.path, draw_opts);
    }
  }

  if( m_layout.get<Gtk::CheckButton>("draw_mst")->get_active()) {
    draw_opts.patPath = gtk::EigenCairo::SolidPattern::create_rgba(
            m_layout.get<Gtk::ColorButton>("mst_color")->get_rgba());
    for (const Node& node : m_graph) {
      Node* parent = node.mst_parent;
      if(!parent) {
        continue;
      }
      for(const Edge& edge : parent->out_edges) {
        if (edge.child == &node) {
          curves_eigen::GenericArtist<double>::draw(parent->state, radius,
                                                    edge.path, draw_opts);
          break;
        }
      }
    }

    for (const Node& node : m_ditch_graph) {
      Node* parent = node.mst_parent;
      if(!parent) {
        continue;
      }
      for(const Edge& edge : parent->out_edges) {
        if (edge.child == &node) {
          curves_eigen::GenericArtist<double>::draw(parent->state, radius,
                                                    edge.path, draw_opts);
          break;
        }
      }
    }

    /* for goal node */ {
      Node* parent = m_goal_node.mst_parent;
      if (parent) {
        for (const Edge& edge : parent->out_edges) {
          if (edge.child == &m_goal_node) {
            curves_eigen::GenericArtist<double>::draw(parent->state, radius,
                                                      edge.path, draw_opts);
            break;
          }
        }
      }
    }
  }

  if (m_layout.get<Gtk::CheckButton>("draw_nominal_plan")->get_active()) {
    draw_opts.patPath = gtk::EigenCairo::SolidPattern::create_rgba(
        m_layout.get<Gtk::ColorButton>("nominal_plan_color")->get_rgba());
    for(const PlanStep& plan_step : m_locked_plan) {
      curves_eigen::GenericArtist<double>::draw(plan_step.start_state, radius,
                                                plan_step.path_to, draw_opts);
    }
  }

  if (m_layout.get<Gtk::CheckButton>("draw_contingency_plan")->get_active()) {
    draw_opts.patPath = gtk::EigenCairo::SolidPattern::create_rgba(
        m_layout.get<Gtk::ColorButton>("contingency_plan_color")->get_rgba());
    for (const Node& ditch_node : m_ditch_graph) {
      const Node* child = &ditch_node;
      const Node* parent = ditch_node.mst_parent;
      while (parent) {
        for (const Edge& edge : parent->out_edges) {
          if (edge.child == child) {
            curves_eigen::GenericArtist<double>::draw(parent->state, radius,
                                                      edge.path, draw_opts);
            break;
          }
        }
        child = parent;
        parent = parent->mst_parent;
      }
    }
  }
}

void Main::LoadObstacles(const YAML::Node& node) {
  for (const auto& point_list : node) {
    m_obstacles.emplace_back();
    for (const auto& point : point_list) {
      m_obstacles.back().emplace_back(
          Eigen::Vector2d(point[0].as<double>(), point[1].as<double>()));
    }
  }
}

void Main::SaveObstacles(YAML::Emitter* yaml_out) {
  (*yaml_out) << YAML::BeginSeq;
  for (auto& point_list : m_obstacles) {
    (*yaml_out) << YAML::BeginSeq;
    for (auto& point : point_list) {
      (*yaml_out) << YAML::BeginSeq << point[0] << point[1] << YAML::EndSeq;
    }
    (*yaml_out) << YAML::EndSeq;
  }
  (*yaml_out) << YAML::EndSeq;
}

void Main::LoadDitchPoints(const YAML::Node& node) {
  for (const auto& point : node) {
    Node graph_node;
    graph_node.state.fill(0);
    graph_node.state[0] = point[0].as<double>();
    graph_node.state[1] = point[1].as<double>();
    m_ditch_graph.push_back(graph_node);
  }
}

void Main::SaveDitchPoints(YAML::Emitter* yaml_out) {
  (*yaml_out) << YAML::BeginSeq;
  for (const Node& node : m_ditch_graph) {
    Eigen::Vector2d point = node.state.head<2>();
    (*yaml_out) << YAML::BeginSeq << point[0] << point[1] << YAML::EndSeq;
  }
  (*yaml_out) << YAML::EndSeq;
}

template <typename T>
int signum(T val) {
  return (T(0) < val) - (val < T(0));
}

/// regardless of the orientation of the polygon, if all the cross products
/// have the same sign then the point is in collision: i.e. the point lies
/// to the "inside" or to the "outside" of all edges. Since the point cannot
/// be to the "outside" of all edges, it must be on the "inside" of all edges.
bool Main::IsCollision(Eigen::Vector2d q) {
  for (auto point_list : m_obstacles) {
    Eigen::Vector3d query_point(0, 0, 0);
    query_point.head<2>() = q;

    Eigen::Vector3d prev_point(0, 0, 0);
    Eigen::Vector3d next_point(0, 0, 0);
    prev_point.head<2>() = point_list.back();
    next_point.head<2>() = point_list.front();

    Eigen::Vector3d prev_product = (query_point - prev_point).cross(
        next_point - prev_point);

    int iter = 0;
    bool this_obstacle_is_collision = true;
    for (const Eigen::Vector2d obstacle_point : point_list) {
      next_point.head<2>() = obstacle_point;
      Eigen::Vector3d current_product = (query_point - prev_point).cross(
          next_point - prev_point);
      ++iter;
      if (signum(current_product[2]) != signum(prev_product[2])) {
        this_obstacle_is_collision = false;
        break;
      }
      prev_product = current_product;
      prev_point.head<2>() = obstacle_point;
    }
    if(this_obstacle_is_collision) {
      return true;
    }
  }
  return false;
}

bool Main::IsCollision(Eigen::Vector3d start_state, Path<double> path) {
  double radius = m_layout.get<Gtk::Adjustment>("vehicle_radius")->get_value();
  double path_length = path.dist(radius);
  double discretization = m_layout.get<Gtk::Adjustment>(
      "collision_discretization")->get_value();
  for (double path_s = 0; path_s <= path_length; path_s += discretization) {
    Eigen::Vector3d state = curves_eigen::IntegrateIncrementally<double>::solve(
        start_state, path, radius, path_s);
    Eigen::Vector2d point = state.head<2>();
    if (IsCollision(point)) {
      return true;
    }
  }
  return false;
}

void Main::GenerateStartEdges() {
  int n_branch =
      m_layout.get<Gtk::Adjustment>("graph_branching_factor")->get_value();
  if (n_branch < 2) {
    n_branch = 2;
  }
  double radius = m_layout.get<Gtk::Adjustment>("vehicle_radius")->get_value();
  if (radius <= 0) {
    radius = 1;
  }

  m_start_node.out_edges.clear();
  for (Node& child : m_graph) {
    Path<double> path =
        curves_eigen::solve(m_start_node.state, child.state, radius);
    if (IsCollision(m_start_node.state, path)) {
      continue;
    }
    // we'll abuse the intention of the 'child' field to connect the parent
    Edge edge{path, path.dist(radius), &child};
    m_start_node.out_edges.insert(edge);
    if (int(m_start_node.out_edges.size()) > n_branch) {
      auto iter = m_start_node.out_edges.end();
      --iter;
      m_start_node.out_edges.erase(iter);
    }
  }
}

void Main::GenerateGraph() {
  Eigen::Vector2d workspace_origin(
      m_layout.get<Gtk::Adjustment>("workspace_x")->get_value(),
      m_layout.get<Gtk::Adjustment>("workspace_y")->get_value());
  Eigen::Vector2d workspace_size(
      m_layout.get<Gtk::Adjustment>("workspace_width")->get_value(),
      m_layout.get<Gtk::Adjustment>("workspace_height")->get_value());

  int n_nodes = m_layout.get<Gtk::Adjustment>("graph_size")->get_value();
  int n_branch =
      m_layout.get<Gtk::Adjustment>("graph_branching_factor")->get_value();
  if(n_branch < 2) {
    n_branch =  2;
  }
  m_graph.clear();
  m_graph.resize(n_nodes);

  // clear out locked path
  m_locked_plan.clear();

  // clear out old mst connections for the goal and ditch points
  m_goal_node.mst_parent = nullptr;
  for (Node& node : m_ditch_graph) {
    node.mst_parent = nullptr;
  }

  // first pass, assign states
  for(Node& node: m_graph) {
    Eigen::Vector2d point;
    bool is_collision = true;
    while(is_collision) {
      point = workspace_origin +
              Eigen::Vector2d(workspace_size[0] * rand() / RAND_MAX,
                              workspace_size[1] * rand() / RAND_MAX);
      is_collision = IsCollision(point);
    }

    node.state.head<2>() = point;
    node.state[2] = (2 * M_PI * rand() / RAND_MAX) - M_PI;
    node.out_edges.clear();
    node.mst_parent = nullptr;
  }

  // second pass, assign connections
  double radius = m_layout.get<Gtk::Adjustment>("vehicle_radius")->get_value();
  if (radius <= 0) {
    radius = 1;
  }
  for (Node& parent : m_graph) {
    // graph interior connections
    for (Node& child : m_graph) {
      Path<double> path = curves_eigen::solve(parent.state, child.state,
                                              radius);
      if (IsCollision(parent.state, path)) {
        continue;
      }
      Edge edge { path, path.dist(radius), &child };
      parent.out_edges.insert(edge);
      if (int(parent.out_edges.size()) > n_branch) {
        auto iter = parent.out_edges.end();
        --iter;
        parent.out_edges.erase(iter);
      }
    }

    // ditch point connections
    for (Node& child : m_ditch_graph) {
      Eigen::Vector2d child_point = child.state.head<2>();
      Path<double> path =
          curves_eigen::PointSolver::solve(parent.state, child_point, radius);
      if (IsCollision(parent.state, path)) {
        continue;
      }
      Edge edge{path, path.dist(radius), &child};
      parent.out_edges.insert(edge);
      if (int(parent.out_edges.size()) > n_branch) {
        auto iter = parent.out_edges.end();
        --iter;
        parent.out_edges.erase(iter);
      }
    }

    // goal point connections
    Eigen::Vector2d goal_point;
    Eigen::Vector2d child_point = m_goal_node.state.head<2>();
    Path<double> path =
        curves_eigen::PointSolver::solve(parent.state, child_point, radius);
    if (IsCollision(parent.state, path)) {
      continue;
    }
    Edge edge{path, path.dist(radius), &m_goal_node};
    parent.out_edges.insert(edge);
    if (int(parent.out_edges.size()) > n_branch) {
      auto iter = parent.out_edges.end();
      --iter;
      parent.out_edges.erase(iter);
    }
  }

  GenerateStartEdges();
  std::cout << "Generated (" << n_nodes << " x " << n_branch << ") graph\n";
  m_view.queue_draw();
}

struct MSTCompare {
  bool operator()(const Node* a, const Node* b) {
    return (a->mst_cost == b->mst_cost) ? (a < b) : (a->mst_cost < b->mst_cost);
  }
};

void Main::BuildMinSpanningTree() {
  double radius = m_layout.get<Gtk::Adjustment>("vehicle_radius")->get_value();
  if (radius <= 0) {
    radius = 1;
  }

  // initialize mst costs and parentage
  for (Node& node : m_graph) {
    node.mst_cost = std::numeric_limits<double>::max();
    node.mst_parent = nullptr;
  }
  for (Node& node : m_ditch_graph) {
    node.mst_cost = std::numeric_limits<double>::max();
    node.mst_parent = nullptr;
  }
  /* for goal node */ {
    m_goal_node.mst_cost = std::numeric_limits<double>::max();
    m_goal_node.mst_parent = nullptr;
  }

  // initialize the open set with the immediate connections from the start
  // point
  std::set<Node*, MSTCompare> open_set;
  std::set<Node*> closed_set;
  m_start_node.mst_cost = 0;
  open_set.insert(&m_start_node);

  // perform Dijkstra's
  while (!open_set.empty()) {
    auto iter = open_set.begin();
    Node* parent = *iter;
    open_set.erase(iter);
    closed_set.insert(parent);
    for (const Edge& edge : parent->out_edges) {
      Node* child = edge.child;
      if (closed_set.count(child)) {
        continue;
      }
      double new_cost = parent->mst_cost + edge.dist;
      if (new_cost < child->mst_cost) {
        open_set.erase(child);
        child->mst_cost = new_cost;
        child->mst_parent = parent;
        open_set.insert(child);
      }
    }
  }
  m_view.queue_draw();
}

void Main::StepAnimation() {
  if(m_locked_plan.size() < 1) {
    return;
  }

  double speed = m_layout.get<Gtk::Adjustment>("vehicle_speed")->get_value();
  double radius = m_layout.get<Gtk::Adjustment>("vehicle_radius")->get_value();
  double dt = m_layout.get<Gtk::Adjustment>("sim_dt")->get_value();
  double path_step = speed * dt;

  m_anim_state += path_step;
  PlanStep& plan_step = m_locked_plan.front();
  if (m_locked_plan.front().path_to.dist(radius) - m_anim_state < 1e-9) {
    m_locked_plan.pop_front();
    m_anim_state = 0;
    m_start_node.state = plan_step.next_node->state;
  } else {
    m_start_node.state = curves_eigen::IntegrateIncrementally<double>::solve(
        plan_step.start_state, plan_step.path_to, radius, m_anim_state);
  }

  GenerateStartEdges();
  BuildMinSpanningTree();
  m_view.queue_draw();
}

void Main::OnSaveImage() {
  Gtk::FileChooserDialog dialog("Please choose a folder",
            Gtk::FILE_CHOOSER_ACTION_SAVE);
  dialog.set_transient_for(*m_layout.get<Gtk::Window>("main"));

  //Add response buttons the the dialog:
  dialog.add_button("_Cancel", Gtk::RESPONSE_CANCEL);
  dialog.add_button("Save", Gtk::RESPONSE_OK);

  int result = dialog.run();

  // Handle the response:
  switch (result) {
    case (Gtk::RESPONSE_OK): {
      std::cout << "File selected: " << dialog.get_filename() << std::endl;
      SaveImage(dialog.get_filename());
      break;
    }
    default: { break; }
  }
}

void Main::SaveImage(const std::string& filename) {
  if(filename.size() < 4) {
    return;
  }
  int width = m_view.get_allocated_width();
  int height = m_view.get_allocated_height();
  std::string suffix = filename.substr(filename.size()-4);
  if (suffix == ".pdf") {
    Cairo::RefPtr<Cairo::PdfSurface> surface =
        Cairo::PdfSurface::create(filename, width, height);
    Cairo::RefPtr<Cairo::Context> context = Cairo::Context::create(surface);
    m_view.on_draw(context);
  } else if (suffix == ".svg") {
    Cairo::RefPtr<Cairo::SvgSurface> surface =
            Cairo::SvgSurface::create(filename, width, height);
    Cairo::RefPtr<Cairo::Context> context = Cairo::Context::create(surface);
    m_view.on_draw(context);
  } else if (suffix == ".png") {
    Cairo::RefPtr<Cairo::ImageSurface> surface =
            Cairo::ImageSurface::create(Cairo::FORMAT_ARGB32, width, height);
    Cairo::RefPtr<Cairo::Context> context = Cairo::Context::create(surface);
    m_view.on_draw(context);
    surface->write_to_png(filename);
  } else {
    std::cerr << "Unrecognized extension for: " << filename << "\n";
  }
}

} // dubins
} // examples
} // mpblocks

namespace ns = mpblocks::examples::dubins;

int main(int argc, char** argv) {
  ns::Main app;
  app.run();
  return 0;
}
