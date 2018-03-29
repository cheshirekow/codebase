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
 *  \file   src/main.cpp
 *
 *  \date   Oct 24, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief
 */

#include <gtkmm.h>

#include <iostream>
#include <Eigen/Dense>
#include <mpblocks/gtk.hpp>
#include <mpblocks/dubins/curves_eigen.hpp>
#include <mpblocks/dubins/curves_eigen/hyper_rect.hpp>
#include <mpblocks/dubins/curves_eigen/cairo.hpp>
#include <mpblocks/util/path_util.h>

namespace mpblocks {
namespace examples {
namespace   dubins {
using namespace mpblocks::dubins;
using namespace mpblocks::dubins::curves_eigen;

typedef Eigen::Matrix<double,3,1> Vector3d_t;

std::string g_stateFile =
    path_util::GetPreferencesPath() + "/dubins_hyper_demo.yaml";


class View : public Gtk::DrawingArea {
 private:
  Eigen::Vector2d scale;

 public:
  sigc::signal<void, const Cairo::RefPtr<Cairo::Context>&> sig_draw;
  sigc::signal<void, double, double> sig_mouseMotion;

  View() {
    add_events(Gdk::POINTER_MOTION_MASK | Gdk::BUTTON_MOTION_MASK |
               Gdk::BUTTON_PRESS_MASK | Gdk::BUTTON_RELEASE_MASK);
    signal_motion_notify_event().connect(
        sigc::mem_fun(*this, &View::on_motion));
  }

  bool on_motion(GdkEventMotion* event) {
    if (event->state & Gdk::BUTTON1_MASK) {
      sig_mouseMotion.emit(event->x / get_allocated_width(),
                           1.0 - (event->y / get_allocated_height()));
    }
    return true;
  }

  virtual void on_size_allocate(Gtk::Allocation& allocation) {
    Gtk::DrawingArea::on_size_allocate(allocation);
    scale[0] = allocation.get_width();
    scale[1] = allocation.get_height();
  }

  virtual bool on_draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
    ctx->rectangle(0, 0, get_allocated_width(), get_allocated_height());
    ctx->set_source_rgb(1, 1, 1);
    ctx->fill_preserve();
    ctx->set_source_rgb(0, 0, 0);
    ctx->stroke();

    ctx->save();
    ctx->scale((double)scale[0], -(double)scale[1]);
    ctx->translate(0, -1);

    sig_draw.emit(ctx);

    ctx->restore();

    return true;
  }
};

class Main {
 private:
  Gtk::Main m_gtkmm;
  gtk::LayoutMap m_layout;
  View m_view;

  Eigen::Vector3d q;
  Eigen::Vector3d min;
  Eigen::Vector3d max;

  Eigen::Vector2d scale;
  Eigen::Vector2d offset;

  double speed;
  double radius;
  double turnRate;

  Eigen::Vector2d cL[4];
  Eigen::Vector2d cR[4];

  // debug storage
  Cairo::RefPtr<Cairo::Pattern> pat_q;
  Cairo::RefPtr<Cairo::Pattern> pat_R;
  Cairo::RefPtr<Cairo::Pattern> pat_L;
  Cairo::RefPtr<Cairo::Pattern> pat_S;

  DrawOpts m_drawOpts;

  int m_iGuiPick;  ///< which solution to draw

 public:
  Main() {
    m_iGuiPick = 0;

    pat_q = Cairo::SolidPattern::create_rgb(0, 0, 0);
    pat_R = Cairo::SolidPattern::create_rgb(0, 0, 1);
    pat_L = Cairo::SolidPattern::create_rgb(0, 0.8, 0);
    pat_S = Cairo::SolidPattern::create_rgb(0.5, 188 / 255.0, 66 / 255.0);
    m_drawOpts.patL = pat_L;
    m_drawOpts.patR = pat_R;
    m_drawOpts.patS = pat_S;
    m_drawOpts.patPath = pat_q;

    loadLayout();
    m_layout.loadValues(g_stateFile);

    setWorkspace();
    setVehicle();
    setStates();
    setPick();

    m_view.sig_draw.connect(sigc::mem_fun(*this, &Main::draw));
    m_view.sig_mouseMotion.connect(sigc::mem_fun(*this, &Main::onMotion));
    sigc::slot<void> wsChanged = sigc::mem_fun(*this, &Main::workspaceChanged);
    sigc::slot<void> qChanged = sigc::mem_fun(*this, &Main::statesChanged);
    sigc::slot<void> vehicleChanged =
        sigc::mem_fun(*this, &Main::vehicleChanged);
    sigc::slot<void> drawChanged = sigc::mem_fun(*this, &Main::drawChanged);
    sigc::slot<void> pickChanged = sigc::mem_fun(*this, &Main::pickChanged);
    sigc::slot<void> constraintsChanged =
        sigc::mem_fun(*this, &Main::constraintsChanged);
    sigc::slot<void> solverChanged = sigc::mem_fun(*this, &Main::solverChanged);
    sigc::slot<void> drawBitsChanged =
        sigc::mem_fun(*this, &Main::drawBitsChanged);

    const char* ws[] = {"ws_x", "ws_y", "ws_w", "ws_h", 0};
    for (const char** adj = ws; *adj != 0; adj++) {
      m_layout.object<Gtk::Adjustment>(*adj)->signal_value_changed().connect(
          wsChanged);
    }

    const char* q[] = {"xMin", "xMax", "q_x",  "yMin", "yMax",
                       "q_y",  "tMin", "tMax", "q_t",  0};
    for (const char** adj = q; *adj != 0; adj++) {
      m_layout.object<Gtk::Adjustment>(*adj)->signal_value_changed().connect(
          qChanged);
    }

    const char* veh[] = {"speed", "min_radius", 0};
    for (const char** adj = veh; *adj != 0; adj++) {
      m_layout.object<Gtk::Adjustment>(*adj)->signal_value_changed().connect(
          vehicleChanged);
    }

    const char* pick[] = {"pick_min", "pick_max", "pick_q", 0};
    for (const char** adj = pick; *adj != 0; adj++) {
      m_layout.widget<Gtk::RadioButton>(*adj)->signal_clicked().connect(
          pickChanged);
    }

    m_layout.widget<Gtk::Button>("saveImage")
        ->signal_clicked()
        .connect(sigc::mem_fun(*this, &Main::saveImage));

    const char* prefix[] = {"x", "y", "theta"};
    const char* suffix[] = {"min", "max", "none"};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        std::stringstream radio;
        radio << "constrain_" << prefix[i] << "_" << suffix[j];
        m_layout.get<Gtk::RadioButton>(radio.str())
            ->signal_clicked()
            .connect(constraintsChanged);
      }
    }
    m_layout.get<Gtk::CheckButton>("constrain_min")
        ->signal_clicked()
        .connect(constraintsChanged);

    for (int i = 0; i < 6; i++) {
      std::stringstream checkBtn;
      checkBtn << "drawBit" << i;
      m_layout.get<Gtk::CheckButton>(checkBtn.str())
          ->signal_clicked()
          .connect(drawBitsChanged);
    }
    m_layout.get<Gtk::CheckButton>("drawBalls")
        ->signal_clicked()
        .connect(drawBitsChanged);

    const char* solvers[] = {"solveLRLa", "solveLRLb", "solveRLRa",
                             "solveRLRb", "solveLSL",  "solveRSR",
                             "solveLSR",  "solveRSL",  "solveMin"};

    for (int i = 0; i < 8; i++) {
      m_layout.get<Gtk::RadioButton>(solvers[i])
          ->signal_clicked()
          .connect(solverChanged);
    }

    solve();
  }

  ~Main() { m_layout.saveValues(g_stateFile); }

 private:
  void loadLayout() {
    std::vector<std::string> paths = {
        path_util::GetSourcePath() + "/src/mpblocks/dubins/hyper/layout.glade",
        path_util::GetResourcePath() + "/dubins_hyper_demo.glade"};

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

    m_layout.widget<Gtk::AspectFrame>("viewFrame")->add(m_view);
    m_view.show();
  }

  void solve() {}

  void setStates() {
    min[0] = m_layout.object<Gtk::Adjustment>("xMin")->get_value();
    min[1] = m_layout.object<Gtk::Adjustment>("yMin")->get_value();
    min[2] = m_layout.object<Gtk::Adjustment>("tMin")->get_value();
    min[2] = clampRadian(min[2] * M_PI / 180.0);

    max[0] = m_layout.object<Gtk::Adjustment>("xMax")->get_value();
    max[1] = m_layout.object<Gtk::Adjustment>("yMax")->get_value();
    max[2] = m_layout.object<Gtk::Adjustment>("tMax")->get_value();
    max[2] = clampRadian(max[2] * M_PI / 180.0);

    q[0] = m_layout.object<Gtk::Adjustment>("q_x")->get_value();
    q[1] = m_layout.object<Gtk::Adjustment>("q_y")->get_value();
    q[2] = m_layout.object<Gtk::Adjustment>("q_t")->get_value();
    q[2] = clampRadian(q[2] * M_PI / 180.0);
  }

  void setWorkspace() {
    scale[0] = m_layout.object<Gtk::Adjustment>("ws_w")->get_value();
    scale[1] = m_layout.object<Gtk::Adjustment>("ws_h")->get_value();

    offset[0] = m_layout.object<Gtk::Adjustment>("ws_x")->get_value();
    offset[1] = m_layout.object<Gtk::Adjustment>("ws_y")->get_value();
  }

  void setVehicle() {
    speed = m_layout.object<Gtk::Adjustment>("speed")->get_value();
    radius = m_layout.object<Gtk::Adjustment>("min_radius")->get_value();
    turnRate = speed / radius;
    m_layout.object<Gtk::Adjustment>("max_rate")
        ->set_value(180 * turnRate / M_PI);
  }

  void setDrawOpt() {}

  void setPick() {
    m_iGuiPick = 0;
    const char* list[] = {"pick_min", "pick_max", "pick_q"};

    for (int i = 0; i < 3; i++) {
      if (m_layout.widget<Gtk::RadioButton>(list[i])->get_active())
        m_iGuiPick = i;
    }
  }

 public:
  void workspaceChanged() {
    setWorkspace();
    m_view.queue_draw();
  }

  void statesChanged() {
    setStates();
    solve();
    m_view.queue_draw();
  }

  void vehicleChanged() {
    setVehicle();
    solve();
    m_view.queue_draw();
  }

  void drawChanged() {
    setDrawOpt();
    m_view.queue_draw();
  }

  void pickChanged() { setPick(); }

  void constraintsChanged() {
    solve();
    m_view.queue_draw();
  }

  void solverChanged() {
    solve();
    m_view.queue_draw();
  }

  void drawBitsChanged() {
    for (int i = 0; i < 6; i++) {
      std::stringstream checkBtn;
      checkBtn << "drawBit" << i;
      m_drawOpts.drawBits[i] =
          m_layout.get<Gtk::CheckButton>(checkBtn.str())->get_active();
    }
    m_drawOpts.drawBalls =
        m_layout.get<Gtk::CheckButton>("drawBalls")->get_active();
    m_view.queue_draw();
  }

  void onMotion(double x, double y) {
    std::stringstream adjX, adjY;
    const char* xList[] = {"xMin", "xMax", "q_x"};
    const char* yList[] = {"yMin", "yMax", "q_y"};

    m_layout.object<Gtk::Adjustment>(xList[m_iGuiPick])
        ->set_value(offset[0] + x * scale[0]);
    m_layout.object<Gtk::Adjustment>(yList[m_iGuiPick])
        ->set_value(offset[1] + y * scale[1]);
  }

  void saveImage() {
    Gtk::FileChooserDialog dialog("Please choose a file",
                                  Gtk::FILE_CHOOSER_ACTION_SAVE);
    dialog.set_transient_for(*m_layout.widget<Gtk::Window>("main"));

    // Add response buttons the the dialog:
    dialog.add_button("Cancel", Gtk::RESPONSE_CANCEL);
    dialog.add_button("Save", Gtk::RESPONSE_OK);

    int result = dialog.run();

    // Handle the response:
    switch (result) {
      case (Gtk::RESPONSE_OK): {
        std::string filename = dialog.get_filename();

        std::cout << "Select clicked." << std::endl;
        std::cout << "File selected: " << filename << std::endl;

        int w = 400;
        int h = 400;
        size_t n = filename.size();
        if (n < 5) {
          std::cout << "invalid filename: " << filename << std::endl;
          return;
        }

        if (strcmp(filename.substr(n - 4, 4).c_str(), ".svg") == 0) {
          Cairo::RefPtr<Cairo::SvgSurface> surf =
              Cairo::SvgSurface::create(filename, w, h);
          Cairo::RefPtr<Cairo::Context> ctx = Cairo::Context::create(surf);
          ctx->scale(w, -h);
          ctx->translate(0, -1);
          draw(ctx);
        }

        if (strcmp(filename.substr(n - 4, 4).c_str(), ".pdf") == 0) {
          Cairo::RefPtr<Cairo::PdfSurface> surf =
              Cairo::PdfSurface::create(filename, w, h);
          Cairo::RefPtr<Cairo::Context> ctx = Cairo::Context::create(surf);
          ctx->scale(w, -h);
          ctx->translate(0, -1);
          draw(ctx);
        }

        if (strcmp(filename.substr(n - 4, 4).c_str(), ".png") == 0) {
          Cairo::RefPtr<Cairo::ImageSurface> surf =
              Cairo::ImageSurface::create(Cairo::FORMAT_ARGB32, w, h);
          Cairo::RefPtr<Cairo::Context> ctx = Cairo::Context::create(surf);
          ctx->scale(w, -h);
          ctx->translate(0, -1);
          draw(ctx);

          surf->write_to_png(filename);
        }
        break;
      }
      default: {
        std::cout << "Unexpected button clicked." << std::endl;
        break;
      }
    }
  }

  void run() {
    drawBitsChanged();
    setDrawOpt();
    setPick();
    setStates();
    setVehicle();
    setWorkspace();
    m_gtkmm.run(*(m_layout.widget<Gtk::Window>("main")));
  }

  struct Constraint {
    double val;
    bool active;

    Constraint() : val(0), active(false) {}

    Constraint& operator=(const double v) {
      val = v;
      active = true;
      return *this;
    }

    operator double&() { return val; }
    operator bool() { return active; }
  };

  void draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
    ctx->scale(1.0 / scale[0], 1.0 / scale[1]);
    ctx->translate(-offset[0], -offset[1]);

    // draw q[0] and q[1];
    ctx->set_source(pat_q);
    ctx->set_line_cap(Cairo::LINE_CAP_ROUND);
    ctx->set_line_width(0.002 * scale.maxCoeff());

    ctx->rectangle(min[0], min[1], (max[0] - min[0]), (max[1] - min[1]));
    ctx->set_source_rgba(1, 0, 0, 0.5);
    ctx->fill();

    typedef Gtk::CheckButton Check;
    typedef Gtk::RadioButton Radio;
    typedef Gtk::Adjustment Adj;

    bool xMin = m_layout.get<Radio>("constrain_x_min")->get_active();
    bool xMax = m_layout.get<Radio>("constrain_x_max")->get_active();
    bool yMin = m_layout.get<Radio>("constrain_y_min")->get_active();
    bool yMax = m_layout.get<Radio>("constrain_y_max")->get_active();
    bool tMin = m_layout.get<Radio>("constrain_theta_min")->get_active();
    bool tMax = m_layout.get<Radio>("constrain_theta_max")->get_active();
    bool minSoln = m_layout.get<Check>("constrain_min")->get_active();

    using namespace hyper;
    uint spec = INVALID_SPEC;
    ;
    if (!minSoln) {
      uint xSpec = boolToSpec(xMin, xMax);
      uint ySpec = boolToSpec(yMin, yMax);
      uint tSpec = boolToSpec(tMin, tMax);
      spec = packSpec(xSpec, ySpec, tSpec);
    }

    // get the solver request
    const char* solvers[] = {"solveLRLa", "solveLRLb", "solveRLRa", "solveRLRb",
                             "solveLSL",  "solveRSR",  "solveLSR",  "solveRSL"};
    for (int i = 0; i < 8; i++) {
      m_drawOpts.solnBits[i] =
          m_layout.get<Gtk::RadioButton>(solvers[i])->get_active();
    }

    ctx->set_source(pat_q);
    Eigen::Vector2d x1, x2, r;
    x1 << q[0], q[1];
    r << std::cos(q[2]), std::sin(q[2]);
    x2 = x1 + speed * r;

    ctx->move_to(x1[0], x1[1]);
    ctx->line_to(x2[0], x2[1]);
    ctx->stroke();

    m_drawOpts.ctx = ctx;
    m_drawOpts.whichSpec = spec;
    DrawOpts opts = m_drawOpts;

    hyper::HyperRect<double> h;
    h.minExt = min;
    h.maxExt = max;

    Path<double> bestSoln = hyper::solve(q, h, radius);
    if (minSoln) {
      std::cout << "best solution is " << bestSoln.dist(radius) << " "
                << "(" << bestSoln.id << (bestSoln.f ? "T" : "F") << ")\n";
      curves_eigen::draw(q, radius, bestSoln, m_drawOpts);
    } else {
      for (int i = 0; i < 8; i++) {
        if (m_drawOpts.solnBits[i]) {
          for (int j = 0; j < 5; j++) {
            if (m_drawOpts.drawBits[j]) {
              std::cout << "Drawing " << spec << "," << i << "," << j << "\n";
              // curves_eigen::draw(q,radius,debugStore(spec,i,j),m_drawOpts);
            }
          }
        }
      }
    }
  }
};

} // dubins
} // examples
} // mpblocks

namespace ns = mpblocks::examples::dubins;

int main( int argc, char** argv )
{
    ns::Main app;
    app.run();
    return 0;
}
