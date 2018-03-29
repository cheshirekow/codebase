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


#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <Eigen/Dense>
#include <gtkmm.h>

#include <mpblocks/dubins/curves_eigen.hpp>
#include <mpblocks/dubins/curves_eigen/cairo.hpp>
#include <mpblocks/gtk.hpp>
#include <mpblocks/util/path_util.h>

#include "algo.h"
#include "cuda_helper.h"
#include "solvers.h"

namespace mpblocks {
namespace examples {
namespace   dubins {
using namespace mpblocks::dubins;

template <typename Scalar>
Scalar getHumanReadable(const Path<Scalar>& path, int j) {
  Scalar mult = 180.0 / M_PI;
  switch (path.id) {
    case LSL:
    case LSR:
    case RSR:
    case RSL:
      if (j == 1) mult = 1.0;
      break;
    default:
      break;
  }

  return mult * path.s[j];
}

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
    ctx->scale(scale[0], -scale[1]);
    ctx->translate(0, -1);

    sig_draw.emit(ctx);

    ctx->restore();

    return true;
  }
};

namespace algo {
const char* radio[] = {"draw_LRLa", "draw_LRLb", "draw_RLRa", "draw_RLRb",
                       "draw_LSL",  "draw_RSR",  "draw_LSR",  "draw_RSL",
                       "draw_min",  0};

const char* buf[] = {"bufLRLa", "bufLRLb", "bufRLRa", "bufRLRb", "bufLSL",
                     "bufRSR",  "bufLSR",  "bufRSL",  "bufMin",  0};
}

class Main {
 private:
  Gtk::Main m_gtkmm;
  gtk::LayoutMap m_layout;
  View m_view;

  Eigen::Vector3d q[2];
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

  Solver* m_solve[INVALID];
  SolutionId m_drawSolver;
  Path<double> m_bestSoln;
  std::map<int,Path<double>> m_solns;
  Solver* m_minSolver;
  int m_iGuiPick;
  SolutionId m_iMinSolver;

  CudaHelper* m_cudaHelper;

 public:
  Main() {
    m_iGuiPick = 0;

    m_cudaHelper = create_cudaHelper();

    pat_q = Cairo::SolidPattern::create_rgb(0, 0, 0);
    pat_R = Cairo::SolidPattern::create_rgb(0, 0, 1);
    pat_L = Cairo::SolidPattern::create_rgb(0, 0.8, 0);
    pat_S = Cairo::SolidPattern::create_rgb(0.5, 188 / 255.0, 66 / 255.0);

    loadLayout();
    m_layout.loadValues(path_util::GetPreferencesPath() +
                        "/dubins_curves.yaml");

    setWorkspace();
    setVehicle();
    setStates();
    setDrawOpt();
    setPick();

    m_view.sig_draw.connect(sigc::mem_fun(*this, &Main::draw));
    m_view.sig_mouseMotion.connect(sigc::mem_fun(*this, &Main::onMotion));
    sigc::slot<void> wsChanged = sigc::mem_fun(*this, &Main::workspaceChanged);
    sigc::slot<void> qChanged = sigc::mem_fun(*this, &Main::statesChanged);
    sigc::slot<void> vehicleChanged =
        sigc::mem_fun(*this, &Main::vehicleChanged);
    sigc::slot<void> drawChanged = sigc::mem_fun(*this, &Main::drawChanged);
    sigc::slot<void> pickChanged = sigc::mem_fun(*this, &Main::pickChanged);

    const char* ws[] = {"ws_x", "ws_y", "ws_w", "ws_h", 0};
    for (const char** adj = ws; *adj != 0; adj++) {
      m_layout.object<Gtk::Adjustment>(*adj)->signal_value_changed().connect(
          wsChanged);
    }

    const char* q[] = {"x1", "y1", "t1", "x2", "y2", "t2", 0};
    for (const char** adj = q; *adj != 0; adj++) {
      m_layout.object<Gtk::Adjustment>(*adj)->signal_value_changed().connect(
          qChanged);
    }

    const char* veh[] = {"speed", "min_radius", 0};
    for (const char** adj = veh; *adj != 0; adj++) {
      m_layout.object<Gtk::Adjustment>(*adj)->signal_value_changed().connect(
          vehicleChanged);
    }

    for (const char** adj = algo::radio; *adj != 0; adj++) {
      m_layout.widget<Gtk::RadioButton>(*adj)->signal_clicked().connect(
          drawChanged);
    }

    const char* pick[] = {"pick1", "pick2", 0};
    for (const char** adj = pick; *adj != 0; adj++) {
      m_layout.widget<Gtk::RadioButton>(*adj)->signal_clicked().connect(
          pickChanged);
    }

    const char* impl[] = {"implTest", "implCuda", "implLib", 0};
    for (const char** radio = impl; *radio != 0; radio++) {
      m_layout.widget<Gtk::RadioButton>(*radio)->signal_clicked().connect(
          qChanged);
    }

    m_layout.widget<Gtk::Button>("saveImage")
        ->signal_clicked()
        .connect(sigc::mem_fun(*this, &Main::saveImage));

    // create solvers
    using namespace algo;
    m_solve[LRLa] = new SolverLRLa();
    m_solve[LRLb] = new SolverLRLb();
    m_solve[RLRa] = new SolverRLRa();
    m_solve[RLRb] = new SolverRLRb();
    m_solve[LSL] = new SolverLSL();
    m_solve[LSR] = new SolverLSR();
    m_solve[RSL] = new SolverRSL();
    m_solve[RSR] = new SolverRSR();

    solve();
  }

  ~Main() {
    m_layout.saveValues(path_util::GetPreferencesPath() +
                        "/dubins_curves.yaml");
    delete m_cudaHelper;
  }

 private:
  void loadLayout() {
    std::vector<std::string> paths = {
        path_util::GetSourcePath() + "/src/mpblocks/dubins/curves/layout.glade",
        path_util::GetResourcePath() + "/dubins_curves_demo.glade"};

    bool layout_loaded = false;
    for( const std::string path : paths ) {
      layout_loaded = m_layout.loadLayout(path);
      if(layout_loaded) {
        break;
      }
    }
    if(!layout_loaded) {
      std::cerr << "Layout not loaded!";
      exit(1);
    }
    m_layout.widget<Gtk::AspectFrame>("viewFrame")->add(m_view);
    m_layout.widget<Gtk::ComboBoxText>("cudaDevices")
        ->signal_changed()
        .connect(sigc::mem_fun(*this, &Main::cudaChanged));
    m_layout.widget<Gtk::TextView>("cudaAttributesV")
        ->override_font(Pango::FontDescription("Monospace 8"));
    m_layout.widget<Gtk::TextView>("cudaPropertiesV")
        ->override_font(Pango::FontDescription("Monospace 8"));
    m_layout.widget<Gtk::TextView>("cudaKernelsV")
        ->override_font(Pango::FontDescription("Monospace 8"));
    m_view.show();
  }

  void solve() {
    if (m_layout.widget<Gtk::RadioButton>("implTest")->get_active()) {
      double distMin = std::numeric_limits<double>::max();
      for (int i = 0; i < INVALID; i++) {
        double dist = m_solve[i]->solve(q, radius);
        if (dist < distMin) {
          m_minSolver = m_solve[i];
          distMin = dist;
        }

        std::stringstream buf;
        buf << std::setprecision(6) << dist;
        m_layout.object<Gtk::EntryBuffer>(algo::buf[i])->set_text(buf.str());

        for (int j = 0; j < 3; j++) {
          std::stringstream adjname;
          adjname << algo::buf[i] << j + 1;
          std::stringstream buf;
          buf << std::setprecision(6) << m_solve[i]->getDist(j);
          m_layout.object<Gtk::EntryBuffer>(adjname.str())->set_text(buf.str());
        }
      }

      std::stringstream buf;
      buf << std::setprecision(6) << distMin;
      m_layout.object<Gtk::EntryBuffer>("distMin")->set_text(buf.str());
    } else if (m_layout.widget<Gtk::RadioButton>("implLib")->get_active()) {
      m_bestSoln = curves_eigen::solve(q[0], q[1], radius);

      m_iMinSolver = (SolutionId)m_bestSoln.id;
      std::stringstream buf;
      buf << std::setprecision(6) << m_bestSoln.dist(radius);
      m_layout.object<Gtk::EntryBuffer>("distMin")->set_text(buf.str());

      for (int i = 0; i < INVALID; i++) {
        Path<double> soln =
            curves_eigen::solve_specific(i, q[0], q[1], radius);
        m_solns[i] = soln;
        if (!soln.f) {
          m_layout.object<Gtk::EntryBuffer>(algo::buf[i])->set_text("");
          for (int j = 0; j < 3; j++) {
            std::stringstream adjname;
            adjname << algo::buf[i] << j + 1;
            m_layout.object<Gtk::EntryBuffer>(adjname.str())->set_text("");
          }
        } else {
          std::stringstream buf;
          buf << std::setprecision(6) << soln.dist(radius);
          m_layout.object<Gtk::EntryBuffer>(algo::buf[i])->set_text(buf.str());

          for (int j = 0; j < 3; j++) {
            std::stringstream adjname;
            adjname << algo::buf[i] << j + 1;
            std::stringstream buf;
            buf << std::setprecision(6)
                << getHumanReadable(soln, j);
            m_layout.object<Gtk::EntryBuffer>(adjname.str())
                ->set_text(buf.str());
          }
        }
      }
    } else if (m_layout.widget<Gtk::RadioButton>("implCuda")->get_active()) {
      m_cudaHelper->solve(m_layout, q[0], q[1], radius);
    }
  }

  void setStates() {
    q[0][0] = m_layout.object<Gtk::Adjustment>("x1")->get_value();
    q[0][1] = m_layout.object<Gtk::Adjustment>("y1")->get_value();
    q[0][2] = m_layout.object<Gtk::Adjustment>("t1")->get_value();
    q[0][2] = M_PI * q[0][2] / 180.0;

    q[1][0] = m_layout.object<Gtk::Adjustment>("x2")->get_value();
    q[1][1] = m_layout.object<Gtk::Adjustment>("y2")->get_value();
    q[1][2] = m_layout.object<Gtk::Adjustment>("t2")->get_value();
    q[1][2] = M_PI * q[1][2] / 180.0;
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

  void setDrawOpt() {
    m_drawSolver = INVALID;
    for (int i = 0; i < INVALID; i++) {
      if (m_layout.widget<Gtk::RadioButton>(algo::radio[i])->get_active()) {
        m_drawSolver = (SolutionId)(i);
      }
    }
  }

  void setPick() {
    for (int i = 0; i < 2; i++) {
      std::stringstream adj;
      adj << "pick" << i + 1;
      if (m_layout.widget<Gtk::RadioButton>(adj.str())->get_active())
        m_iGuiPick = i;
    }
  }

 public:
  void cudaChanged() { m_cudaHelper->populateDetails(m_layout); }

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

  void onMotion(double x, double y) {
    std::stringstream adjX, adjY;
    adjX << 'x' << m_iGuiPick + 1;
    adjY << 'y' << m_iGuiPick + 1;

    m_layout.object<Gtk::Adjustment>(adjX.str())
        ->set_value(offset[0] + x * scale[0]);
    m_layout.object<Gtk::Adjustment>(adjY.str())
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
    m_cudaHelper->populateDevices(m_layout);
    cudaChanged();
    solve();
    m_gtkmm.run(*(m_layout.widget<Gtk::Window>("main")));
  }

  void draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
    ctx->scale(1.0 / scale[0], 1.0 / scale[1]);
    ctx->translate(-offset[0], -offset[1]);

    // draw q[0] and q[1] regardless of solver implementation
    ctx->set_source(pat_q);
    ctx->set_line_cap(Cairo::LINE_CAP_ROUND);
    ctx->set_line_width(0.002 * scale.maxCoeff());

    Eigen::Vector2d x1, x2, r;
    x1 << q[0][0], q[0][1];
    r << std::cos(q[0][2]), std::sin(q[0][2]);
    x2 = x1 + speed * r;

    ctx->move_to(x1[0], x1[1]);
    ctx->line_to(x2[0], x2[1]);
    ctx->stroke();

    x1 << q[1][0], q[1][1];
    r << std::cos(q[1][2]), std::sin(q[1][2]);
    x2 = x1 + speed * r;

    ctx->move_to(x1[0], x1[1]);
    ctx->line_to(x2[0], x2[1]);
    ctx->stroke();

    if (m_layout.widget<Gtk::RadioButton>("implTest")->get_active()) {
      if (m_drawSolver == INVALID) {
        if (m_minSolver) m_minSolver->draw(ctx);
      } else
        m_solve[m_drawSolver]->draw(ctx);
    } else if (m_layout.widget<Gtk::RadioButton>("implLib")->get_active()) {
      curves_eigen::DrawOpts opts;
      opts.ctx = ctx;
      for (int i = 0; i < 2; i++) opts.dash[i] = 0.01 * scale.maxCoeff();
      opts.patL = pat_L;
      opts.patR = pat_R;
      opts.patS = pat_S;
      opts.patPath = pat_q;

      if (m_drawSolver == INVALID) {
        curves_eigen::draw(q[0], radius, m_bestSoln, opts);
      } else {
        curves_eigen::draw(q[0], radius, m_solns[m_drawSolver], opts);
      }
    }
  }
};

} // dubins
} // examples
} // mpblocks

namespace ns = mpblocks::examples::dubins;

int main(int argc, char** argv) {
  ns::Main app;
  app.run();
  return 0;
}
