/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of convex_hull.
 *
 *  convex_hull is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  convex_hull is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Fontconfigmm.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   main.cpp
 *
 *  @date   Aug 15, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */
#include <functional>

#include <mpblocks/brown79.hpp>
#include <mpblocks/clarkson93.hpp>
#include <mpblocks/gtk.hpp>
#include <mpblocks/btps.h>

#include <gtkmm.h>
#include <boost/format.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <functional>

#include "config.h"

using namespace mpblocks;

/// an example traits structure from which a balanced tree of partial sums
/// may be instantiated
struct BTPSTraits {
  /// need to forward declare Node so we can make NodeRef be a pointer
  struct Node;

  /// some type which stores uniquely identifies nodes, for instance
  /// a node pointer or an index into an array
  typedef Node* NodeRef;

  /// this type is not required by the interface, but if you just need a
  /// simple node type then this one will do it for you
  struct Node : btps::BasicNode<BTPSTraits> {
    Node(double weight = 0) : BasicNode<BTPSTraits>(weight) {}
  };

  /// a callable type which implements the primitives required to access
  /// fields of a node
  struct NodeOps {
    /// you can leave this one off you if you dont need removal
    NodeRef& Parent(NodeRef N) { return N->parent; }

    /// return the left child of N
    NodeRef& LeftChild(NodeRef N) { return N->left; }

    /// return the right child of N
    NodeRef& RightChild(NodeRef N) { return N->right; }

    /// return the weight of this node in particular
    double Weight(NodeRef N) { return N->weight; }

    /// return the cumulative weight of the subtree, the return type
    /// is deduced by the tree template and can be anything modeling a
    /// real number
    double& CumulativeWeight(NodeRef N) { return N->cumweight; }

    /// return the subtree node count
    uint32_t& Count(NodeRef N) { return N->count; }
  };
};

typedef BTPSTraits::Node        BTPSNode;
typedef btps::Tree<BTPSTraits>  BTPSTree;

struct ClarksonTraits {
  typedef unsigned int uint;

  /// dimension of the embeded space, use clarkson93::Dynamic for a
  /// datastructure whose dimension is set at runtime
  static const unsigned int NDim = 3;

  /// number format for scalar numbers
  typedef double Scalar;

  /// Data structure for representing points.
  /**
   *  Currently only
   *  Eigen::Matrix<Scalar,NDim,1> is supported b/c we utilize this structure
   *  in solving a linear system. Technically anything that derives from
   *  Eigen::MatrixBase will work because it just needs to be assignable to
   *  an Eigen::Matrix::row() and from an Eigen::Vector when we solve for
   *  the normal and offset of the base facet. In the future I may generalize
   *  the solving of the linear system in which case that functionality will
   *  be a requirement of the traits class
   */
  typedef Eigen::Matrix<Scalar, NDim, 1> Point;

  // forward declare so we can define SimplexRef
  struct Simplex;

  /// a reference to a point
  typedef uint PointRef;

  /// a reference to a simplex
  typedef Simplex* SimplexRef;

  /// proves a means of turning a PointRef into a Point&
  /**
   *  In the example traits PointRef is a Point* so we simply dereference this
   *  pointer. If PointRef were an index into an array of Point structures,
   *  then PointDeref should store a pointer to the beginning of that array.
   *  For example
   *
   *  @code
   typedef unsigned int PointRef;
   struct PointDeref
   {
   Point* m_buf;
   Point& operator()( PointRef i ){ return m_buf[i]; }
   };
   @endcode
   */
  class Deref {
   private:
    std::vector<Point>* m_buf;

   public:
    Deref(std::vector<Point>* buf = 0) { setBuf(buf); }
    void setBuf(std::vector<Point>* buf) { m_buf = buf; }

    Point& point(PointRef idx) { return (*m_buf)[idx]; }
    Simplex& simplex(SimplexRef ptr) { return *ptr; }
  };

  /// The derived type to use for simplices.
  /**
   *  The only requirement is that it
   *  must derive from clarkson93::SimplexBase instantiated with the Traits
   *  class. Otherwise, it may contain as many extra data members as you want.
   *  You may hook into the triangulation control flow by overriding
   *  member functions of SimplexBase. If you do so your implementation will
   *  be called, instead of SimplexBase's (even though your implementation is
   *  not virtual). This is because the triangulation code maintains only
   *  Simplex* pointers (i.e. pointers to this type) not pointers to the
   *  base type. If you choose to hook into the program flow this way then
   *  you must be sure to call the SimplexBase member functions from your
   *  override.
   */
  struct Simplex : clarkson93::Simplex2<ClarksonTraits>, BTPSNode {
    typedef clarkson93::Simplex2<ClarksonTraits> Base;

    Eigen::Matrix<double, 2, 1> c_c;  ///< circum center
    double c_r;                       ///< circum radius
    bool c_has;                       ///< has center

    Simplex() : Base(-1, 0), c_r(0), c_has(false) {}
  };

  /// Triangulation will derive from this in order to inherit operations
  /// for simplices
  typedef clarkson93::SimplexOps<ClarksonTraits> SimplexOps;

  /// template for allocators & memory managers
  /**
   *  The triangulation doesn't support removal of points. Thus simplices are
   *  never deleted individually (rather, all at once when the triangulation
   *  is cleared). Thus we divert memory management to the user of the
   *  libary. The allocator is responsible for keeping track of every object
   *  it allocates an be delete them all when cleared
   *
   *  This example preallocates a fixed size array of storage for objects.
   *  When create() is called it takes the next unused pointers, calls
   *  in-place operator new() and then returns the pointer. When it is
   *  reset, it goes through the list of allocated pointers and calls the
   *  destructor (ptr)->~T() on each of them.
   *
   *  For the most part, Alloc<T> is only used to construct POD types so
   *  this is a bit of a moot point.
   *
   *  @note: Alloc must be default constructable. It will be passed to
   *         setup.prepare( Alloc<T> ) so if you need to do any allocator
   *         initialization you must do it in Setup
   */
  struct SimplexMgr : public std::vector<Simplex> {
    typedef std::vector<Simplex> base_t;

    /// construct an object of type T using the next available memory
    // slot and return it's pointer
    SimplexRef create() {
      assert(base_t::size() < base_t::capacity());
      base_t::emplace_back();
      return &base_t::back();
    }
  };

  /// the triangulation provides some static callbacks for when hull faces
  /// are added or removed. If we wish to do anything special this is where
  /// we can hook into them. If you do not wish to hook into the callbacks
  /// then simply create an empty structure which has empty implementation
  /// for these
  struct Callback {
    std::function<void(Simplex*)> sig_hullFaceAdded;
    std::function<void(Simplex*)> sig_hullFaceRemoved;

    void hullFaceAdded(Simplex* S) { sig_hullFaceAdded(S); }

    void hullFaceRemoved(Simplex* S) { sig_hullFaceRemoved(S); }
  };
};

typedef ClarksonTraits::Simplex     Simplex;
typedef ClarksonTraits::Point       Point;
typedef ClarksonTraits::PointRef    PointRef;
typedef ClarksonTraits::SimplexRef  SimplexRef;
typedef ClarksonTraits::Deref       Deref;

typedef clarkson93::Triangulation<ClarksonTraits>   Triangulation;
typedef brown79::Inversion<ClarksonTraits>          Inversion;

/// calculates vertices of an arrow-head to draw at the end of a vector
/// from start to end
void calcVertexes(const Point& start, const Point& end, Point& p1, Point& p2,
                  double length, double degrees) {
  double end_y = end[1];
  double start_y = start[1];
  double end_x = end[0];
  double start_x = start[0];

  double angle = atan2(end_y - start_y, end_x - start_x) + M_PI;

  p1[0] = end_x + length * cos(angle - degrees);
  p1[1] = end_y + length * sin(angle - degrees);
  p2[0] = end_x + length * cos(angle + degrees);
  p2[1] = end_y + length * sin(angle + degrees);
}

template <typename T>
constexpr T factoral(T n) {
  return n == 1 ? 1 : n * factoral(n - 1);
}

const int nFactoral = factoral(2);

double measure(Inversion& inv, Deref& deref, Simplex* S) {
  typedef Eigen::Matrix<double, 2, 2> Matrix;
  typedef Eigen::Matrix<double, 2, 1> Vector;

  PointRef peak = S->V[S->iPeak];
  std::vector<PointRef> pRefs;
  std::copy_if(S->V, S->V + 4, std::back_inserter(pRefs),
               [peak](PointRef v) { return peak != v; });

  Matrix A;
  Vector x0 = inv(deref.point(pRefs[0])).block(0, 0, 2, 1);
  for (int i = 1; i < 3; i++) {
    Vector xi = inv(deref.point(pRefs[i])).block(0, 0, 2, 1);
    A.row(i - 1) = (xi - x0);
  }

  return std::abs(A.determinant() / nFactoral);
}

void computeCenter(Inversion& inv, Deref& deref, Simplex* S) {
  typedef Eigen::Matrix<double, 2, 2> Matrix;
  typedef Eigen::Matrix<double, 2, 1> Vector;

  // calculate circumcenter of the first simplex
  // see 2012-10-26-Note-11-41_circumcenter.xoj for math
  Matrix A;
  Vector b;

  PointRef peak = S->V[S->iPeak];
  std::vector<PointRef> pRefs;
  std::copy_if(S->V, S->V + 4, std::back_inserter(pRefs),
               [peak](PointRef v) { return peak != v; });

  Vector x0 = inv(deref.point(pRefs[0])).block(0, 0, 2, 1);
  for (int i = 1; i < 3; i++) {
    Vector xi = inv(deref.point(pRefs[i])).block(0, 0, 2, 1);
    Vector dv = 2 * (xi - x0);
    A.row(i - 1) = dv;
    b[i - 1] = xi.squaredNorm() - x0.squaredNorm();
  }

  // the circum center
  S->c_c = A.fullPivLu().solve(b);

  // squared radius of the circumsphere
  S->c_r = (S->c_c - x0).norm();
  S->c_has = true;
}

struct Main {
  std::string m_layoutFile;
  std::string m_stateFile;

  Gtk::Main m_gtkmm;
  gtk::LayoutMap m_layout;
  gtk::SimpleView m_view;

  sigc::connection auto_cnx;

  Triangulation m_T;
  Inversion m_inv;
  std::vector<Point> m_ptStore;

  BTPSTree m_btps;
  BTPSNode m_btpsNil;

  std::map<int, std::list<BTPSNode*> > m_btpsProfile;

  Main() : m_btps(&m_btpsNil) {
    m_layoutFile = std::string(g_srcDir) + "/layout.glade";
    m_stateFile = std::string(g_binDir) + "/state.yaml";

    m_layout.loadLayout(m_layoutFile);
    m_layout.loadValues(m_stateFile);

    m_layout.widget<Gtk::AspectFrame>("viewFrame")->add(m_view);
    m_view.show();

    sigc::slot<void, const Cairo::RefPtr<Cairo::Context>&> slot_draw =
        sigc::mem_fun(*this, &Main::draw);
    sigc::slot<void, double, double> slot_addPoint =
        sigc::mem_fun(*this, &Main::addPoint);
    sigc::slot<void, double, double> slot_onMove =
        sigc::mem_fun(*this, &Main::onMovement);

    m_view.sig_draw.connect(slot_draw);
    m_view.sig_motion.connect(slot_onMove);
    m_view.sig_press.connect(slot_addPoint);

    m_layout.get<Gtk::Button>("btn_clear")
        ->signal_clicked()
        .connect(sigc::mem_fun(this, &Main::clear));

    const char* keys[] = {"chk_sites",   "chk_delaunay", "chk_balls",
                          "chk_vpoints", "chk_edges",    "chk_weights"};

    for (int i = 0; i < 5; i++) {
      m_layout.get<Gtk::CheckButton>(keys[i])->signal_clicked().connect(
          sigc::mem_fun(m_view, &gtk::SimpleView::queue_draw));
    }

    m_layout.get<Gtk::ToggleButton>("tbtn_auto")
        ->signal_toggled()
        .connect(sigc::mem_fun(this, &Main::autoStateChanged));

    m_layout.get<Gtk::Button>("btn_random")
        ->signal_clicked()
        .connect(sigc::mem_fun(this, &Main::random));
    m_layout.get<Gtk::Button>("btn_random_n")
        ->signal_clicked()
        .connect(sigc::mem_fun(this, &Main::onRandom_n));
    m_layout.get<Gtk::Button>("btn_clear")
        ->signal_clicked()
        .connect(sigc::mem_fun(this, &Main::clear));
    m_layout.get<Gtk::Button>("saveImage")
        ->signal_clicked()
        .connect(sigc::mem_fun(this, &Main::saveImage));

    m_layout.get<Gtk::Adjustment>("edgeWidth")
        ->signal_value_changed()
        .connect(sigc::mem_fun(m_view, &gtk::SimpleView::queue_draw));
    m_layout.get<Gtk::Adjustment>("pointRadius")
        ->signal_value_changed()
        .connect(sigc::mem_fun(m_view, &gtk::SimpleView::queue_draw));

    // monospace font
    Pango::FontDescription fontdesc("monospace");
    m_layout.widget<Gtk::TextView>("depthView")->override_font(fontdesc);

    m_inv.init(Point(0.5, 0.5, 1), 1);
    m_ptStore.reserve(100000);
    m_T.m_sMgr.reserve(100000);
    m_T.m_antiOrigin = -1;
    m_T.m_xv_walked.reserve(1000);
    m_T.m_xv_queue.reserve(1000);
    m_T.m_xvh.reserve(1000);
    m_T.m_xvh_queue.reserve(1000);
    m_T.m_ridges.reserve(1000);
    m_T.m_deref.setBuf(&m_ptStore);

    using namespace std::placeholders;
    m_T.m_callback.sig_hullFaceAdded = std::bind(&Main::simplexAdded, this, _1);
    m_T.m_callback.sig_hullFaceRemoved =
        std::bind(&Main::simplexRemoved, this, _1);
  }

  ~Main() { m_layout.saveValues(m_stateFile); }

  void simplexAdded(Simplex* S) {
    if (m_T.isVisible(*S, m_inv.center())) return;

    S->weight = measure(m_inv, m_T.m_deref, S);
    m_btps.insert(S);

    //            std::cout << "Added simplex with measure " << S-> weight
    //                      << " total is now: " << m_btps.sum()
    //                      << " nil has weight " << m_btpsNil.weight
    //                      << ", cum: " << m_btpsNil.cumweight<< "\n";
    //            m_hullSize++;
    //            std::cout << "add, size:" << m_btps.size() << ", should be "
    //                      << m_hullSize << "\n";
  }

  void simplexRemoved(Simplex* S) {
    if (m_T.isVisible(*S, m_inv.center())) return;

    m_btps.remove(S);
    S->parent = 0;
    S->right = 0;
    S->left = 0;

    //            std::cout << "Removed simplex with measure " << S-> weight
    //                      << " total is now: " << m_btps.sum()
    //                      << " nil has weight " << m_btpsNil.weight
    //                      << ", cum: " << m_btpsNil.cumweight<< "\n";
    //            m_hullSize--;
    //            std::cout << "remove, size:" << m_btps.size() << ", should be
    //            "
    //                      << m_hullSize << "\n";
  }

  void onMovement(double x, double y) {
    m_layout.get<Gtk::Adjustment>("adj_x")->set_value(x);
    m_layout.get<Gtk::Adjustment>("adj_y")->set_value(y);
  }

  void addPoint(double x, double y) {
    m_ptStore.push_back(m_inv(Point(x, y, 0)));

    if (m_ptStore.size() < 4) {
    } else if (m_ptStore.size() == 4)
      m_T.init(0, 4, [](uint i) { return i; });
    else {
      PointRef idx = m_ptStore.size() - 1;
      m_T.insert(idx);
    }
    //            std::cout << "Total measure: " << m_btps.sum() << "\n";

    m_view.queue_draw();
  }

  void clear() {
    m_T.clear();
    m_btps.clear();
    m_ptStore.clear();
    m_btpsProfile.clear();
    m_view.queue_draw();
  }

  void run() { m_gtkmm.run(*m_layout.widget<Gtk::Window>("main")); }

  void profileBTPS() {
    m_btpsProfile.clear();
    m_btps.generateDepthProfile(m_btpsProfile);

    typedef boost::format fmt;

    std::stringstream report;
    for (auto& pair : m_btpsProfile)
      report << fmt(" %4u : %4u \n") % pair.first % pair.second.size();

    m_layout.get<Gtk::TextBuffer>("depthReport")->set_text(report.str());
  }

  void random_n(int n) {
    if (m_layout.get<Gtk::RadioButton>("sampleCube")->get_active())
      random_cube_n(n);
    else
      random_interior_n(n);
    if (m_layout.get<Gtk::CheckButton>("profileBTPS")->get_active())
      profileBTPS();
  }

  void onRandom_n() {
    random_n(m_layout.get<Gtk::Adjustment>("randBatch")->get_value());
  }

  void random() {
    if (m_layout.get<Gtk::RadioButton>("sampleCube")->get_active())
      random_cube();
    else
      random_interior();
    if (m_layout.get<Gtk::CheckButton>("profileBTPS")->get_active())
      profileBTPS();
  }

  void random_cube_n(int n) {
    for (int i = 0; i < n; i++) random_cube();
  }

  void random_cube() {
    addPoint(rand() / (double)RAND_MAX, rand() / (double)RAND_MAX);
  }

  void random_interior_n(int n) {
    for (int i = 0; i < n; i++) random_interior();
  }

  void random_interior() {
    // can't generate without at least one simplex
    if (m_ptStore.size() < 4) {
      addPoint(rand() / (double)RAND_MAX, rand() / (double)RAND_MAX);
      return;
    }

    // select a node from the weighted disribution
    BTPSNode* node = m_btps.findInterval(rand() / (double)RAND_MAX);

    // cast it to a simplex
    Simplex* S = static_cast<Simplex*>(node);

    // generate a random barycentric coordinate
    std::set<double> Z;
    for (int i = 0; i < 2; i++) Z.insert(rand() / (double)RAND_MAX);
    Z.insert(1);

    std::vector<double> lambda;
    double z_prev = 0;
    for (double z : Z) {
      lambda.push_back(z - z_prev);
      z_prev = z;
    }

    // compute the point at that coordinate
    PointRef peak = m_T.peak(*S);
    std::vector<PointRef> pts;
    std::copy_if(S->V, S->V + 4, std::back_inserter(pts),
                 [peak](PointRef q) { return q != peak; });

    Point p = lambda[0] * m_inv(m_ptStore[pts[0]]) +
              lambda[1] * m_inv(m_ptStore[pts[1]]) +
              lambda[2] * m_inv(m_ptStore[pts[2]]);
    //            std::cout << "lambda: "
    //                      << lambda [0] << " + "
    //                      << lambda [1] << " + "
    //                      << lambda [2] << " = "
    //                        << lambda[0] + lambda[1] + lambda[2] << "\n";
    p[2] = 0;

    // insert that point into the triangulation
    m_ptStore.push_back(m_inv(p));
    m_T.insert(m_ptStore.size() - 1, S);

    // update the view
    m_view.queue_draw();
  }

  bool onSignal() {
    random_n(m_layout.get<Gtk::Adjustment>("autoRate")->get_value());
    return true;
  }

  void autoStateChanged() {
    if (auto_cnx.empty()) {
      auto_cnx = Glib::signal_timeout().connect(
          sigc::mem_fun(*this, &Main::onSignal), 10);
    } else
      auto_cnx.disconnect();
  }

  void draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
    double eWidth =
        m_layout.get<Gtk::Adjustment>("edgeWidth")->get_value() / 20000.0;
    double rPoint =
        m_layout.get<Gtk::Adjustment>("pointRadius")->get_value() / 10000.0;

    gtk::EigenCairo ectx(ctx);
    ctx->set_line_width(eWidth);

    if (m_layout.get<Gtk::CheckButton>("chk_sites")->get_active()) {
      Gdk::RGBA color =
          m_layout.get<Gtk::ColorButton>("color_sites")->get_rgba();
      ctx->set_source_rgba(color.get_red(), color.get_green(), color.get_blue(),
                           color.get_alpha());
      for (const Point& p : m_ptStore) {
        Point x0 = m_inv(p);
        Eigen::Matrix<double, 2, 1> x1(x0[0], x0[1]);
        ectx.circle(x1, rPoint);
        ctx->fill();
      }
    }

    if (m_ptStore.size() < 4) return;

    bool drawDelaunay =
        m_layout.get<Gtk::CheckButton>("chk_delaunay")->get_active();
    bool drawWeights =
        m_layout.get<Gtk::CheckButton>("chk_weights")->get_active();
    bool drawCircumCenter =
        m_layout.get<Gtk::CheckButton>("chk_vpoints")->get_active();
    bool drawCircumSphere =
        m_layout.get<Gtk::CheckButton>("chk_balls")->get_active();
    bool drawVoronoi =
        m_layout.get<Gtk::CheckButton>("chk_edges")->get_active();

    // pointer to the point buffer
    Point* pts = m_ptStore.data();
    Deref deref(&m_ptStore);

    for (Simplex& Sref : m_T.m_sMgr) {
      Simplex* S = &Sref;

      if (!Sref.sets[clarkson93::simplex::HULL]) continue;
      if (m_T.isVisible(Sref, m_inv.center())) continue;

      // it is a nearest site if the inversion center is not
      // visible, so draw it
      if (drawDelaunay || drawWeights) {
        PointRef peak = S->V[S->iPeak];
        std::vector<PointRef> pRefs;
        std::copy_if(S->V, S->V + 4, std::back_inserter(pRefs),
                     [peak](PointRef v) { return peak != v; });

        Point x0 = m_inv(pts[pRefs[2]]);
        ctx->move_to((double)x0[0], (double)x0[1]);
        for (int i = 0; i < 3; i++) {
          Point x1 = m_inv(pts[pRefs[i]]);
          ctx->line_to((double)x1[0], (double)x1[1]);
        }

        if (drawWeights) {
          Gdk::RGBA color =
              m_layout.get<Gtk::ColorButton>("color_weights")->get_rgba();
          ctx->set_source_rgba(color.get_red(), color.get_green(),
                               color.get_blue(), S->weight / m_btps.sum());
          ctx->fill_preserve();
        }

        if (drawDelaunay) {
          Gdk::RGBA color =
              m_layout.get<Gtk::ColorButton>("color_delaunay")->get_rgba();
          ctx->set_source_rgba(color.get_red(), color.get_green(),
                               color.get_blue(), color.get_alpha());
          ctx->stroke();
        }
      }

      computeCenter(m_inv, deref, S);

      if (drawCircumCenter || drawCircumSphere) {
        if (drawCircumSphere) {
          Gdk::RGBA color =
              m_layout.get<Gtk::ColorButton>("color_balls")->get_rgba();
          ctx->set_source_rgba(color.get_red(), color.get_green(),
                               color.get_blue(), color.get_alpha());
          ectx.circle(S->c_c, S->c_r);
          ctx->fill();
        }

        if (drawCircumCenter) {
          Gdk::RGBA color =
              m_layout.get<Gtk::ColorButton>("color_vpoints")->get_rgba();
          ctx->set_source_rgba(color.get_red(), color.get_green(),
                               color.get_blue(), color.get_alpha());
          ectx.circle(S->c_c, rPoint);
          ctx->fill();
        }
      }

      if (drawVoronoi) {
        for (int i = 1; i < 4; i++) {
          Simplex* N = S->N[i];
          if (!m_T.isVisible(*N, m_inv.center())) {
            computeCenter(m_inv, deref, N);
            Gdk::RGBA color =
                m_layout.get<Gtk::ColorButton>("color_edges")->get_rgba();
            ctx->set_source_rgba(color.get_red(), color.get_green(),
                                 color.get_blue(), color.get_alpha());

            ectx.move_to(S->c_c);
            ectx.line_to(N->c_c);
            ctx->stroke();
          }
        }
      }
    }
  }

  void saveImage() {
    Gtk::FileChooserDialog dialog("Save Image", Gtk::FILE_CHOOSER_ACTION_SAVE);
    dialog.set_transient_for(*m_layout.get<Gtk::Window>("main"));

    // Add response buttons the the dialog:
    dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
    dialog.add_button(Gtk::Stock::OPEN, Gtk::RESPONSE_OK);

    // Add filters, so that only certain file types can be selected:

    Glib::RefPtr<Gtk::FileFilter> filter;

    filter = Gtk::FileFilter::create();
    filter->set_name("Portable Network Graphics (png)");
    filter->add_mime_type("image/png");
    dialog.add_filter(filter);

    filter = Gtk::FileFilter::create();
    filter->set_name("Portable Document Format (pdf)");
    filter->add_mime_type("application/x-pdf");
    dialog.add_filter(filter);

    filter = Gtk::FileFilter::create();
    filter->set_name("Scalable Vector Graphics (svg)");
    filter->add_mime_type("image/svg-xml");
    dialog.add_filter(filter);

    // Show the dialog and wait for a user response:
    int result = dialog.run();

    // Handle the response:
    switch (result) {
      case (Gtk::RESPONSE_OK): {
        std::string filename = dialog.get_filename();
        int n = filename.size();
        if (n < 5) {
          std::cout << "invalid filename: " << filename << std::endl;
          return;
        }

        std::string ext = filename.substr(n - 4, 4);
        if (ext.compare(".png") == 0) {
          int w = 800;
          int h = 800;

          Cairo::RefPtr<Cairo::ImageSurface> img =
              Cairo::ImageSurface::create(Cairo::FORMAT_ARGB32, w, h);
          Cairo::RefPtr<Cairo::Context> ctx = Cairo::Context::create(img);

          ctx->scale(w, -h);
          ctx->translate(0, -1);
          ctx->set_source_rgb(1, 1, 1);
          ctx->paint();

          draw(ctx);

          img->write_to_png(filename);
        } else if (ext.compare(".pdf") == 0) {
          int w = 400;
          int h = 400;
          Cairo::RefPtr<Cairo::PdfSurface> img =
              Cairo::PdfSurface::create(filename, w, h);
          Cairo::RefPtr<Cairo::Context> ctx = Cairo::Context::create(img);

          ctx->scale(w, -h);
          ctx->translate(0, -1);
          ctx->set_source_rgb(1, 1, 1);
          ctx->paint();

          draw(ctx);
        } else if (ext.compare(".svg") == 0) {
          int w = 400;
          int h = 400;
          Cairo::RefPtr<Cairo::SvgSurface> img =
              Cairo::SvgSurface::create(filename, w, h);
          Cairo::RefPtr<Cairo::Context> ctx = Cairo::Context::create(img);

          ctx->scale(w, -h);
          ctx->translate(0, -1);
          ctx->set_source_rgb(1, 1, 1);
          ctx->paint();

          draw(ctx);
        } else {
          std::cout << "invalid file format: " << filename << std::endl;
          return;
        }

        break;
      }
    }
  }
};

int main(int argc, char** argv) {
  Main m_app;
  m_app.run();
  return 0;
}
