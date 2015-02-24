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
 *  \file   main.cpp
 *
 *  \date   Aug 15, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <mpblocks/brown79.hpp>
#include <mpblocks/clarkson93.hpp>
#include <mpblocks/gtk.hpp>

#include <cstdlib>
#include <gtkmm.h>
#include <iostream>
#include <string>

#include "config.h"

class View: public Gtk::DrawingArea
{
    public:
        sigc::signal<void, const Cairo::RefPtr<Cairo::Context>&> sig_draw;
        sigc::signal<void, double, double> sig_mouse;
        sigc::signal<void, double, double> sig_click;

        View()
        {
            this->add_events(Gdk::POINTER_MOTION_MASK | Gdk::BUTTON_PRESS_MASK);

            this->signal_button_press_event().connect(
                    sigc::mem_fun(*this, &View::on_click));
            this->signal_motion_notify_event().connect(
                    sigc::mem_fun(*this, &View::on_move));
        }

        bool on_move(GdkEventMotion* evt)
        {
            double x = evt->x / this->get_allocated_width();
            double y = 1.0 - evt->y / this->get_allocated_height();
            sig_mouse(x, y);
            return true;
        }

        bool on_click(GdkEventButton* evt)
        {
            if (evt->button == 1)
            {
                double x = evt->x / this->get_allocated_width();
                double y = 1.0 - evt->y / this->get_allocated_height();
                sig_click(x, y);
            }
            return true;
        }

        virtual bool on_draw(const Cairo::RefPtr<Cairo::Context>& ctx)
        {
            int width = this->get_allocated_width();
            int height = this->get_allocated_height();

            // draw white background
            ctx->set_source_rgb(1, 1, 1);
            ctx->paint();

            // scale the context so (0,0) is bottom left and (1,1) is top
            // right
            ctx->scale(width, -height);
            ctx->translate(0, -1);

            sig_draw.emit(ctx);

            return true;
        }
};

using namespace mpblocks;

/// documents the interface for Traits : encapsulates various policies for the
/// Triangulation
/**
 *  @todo   Should these traits be split up into multiple policy classes?
 */
struct ClarksonTraits
{
        typedef unsigned int uint;

        /// dimension of the embeded space, use clarkson93::Dynamic for a
        /// datastructure whose dimension is set at runtime
        static const unsigned int NDim = 3;

        /// number format for storing indices
        typedef unsigned int idx_t;

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

        /// a reference to a point
        typedef uint PointRef;

        /// acts like a uint but has distinct type
        struct ConstPointRef
        {
                uint m_storage;
                ConstPointRef(uint in = 0) :
                        m_storage(in)
                {
                }
                operator uint&()
                {
                    return m_storage;
                }
        };

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
        class PointDeref
        {
            private:
                Point* m_buf;

            public:
                PointDeref(Point* buf = 0)
                {
                    setBuf(buf);
                }
                void setBuf(Point* buf)
                {
                    m_buf = buf;
                }
                Point& operator()(PointRef idx)
                {
                    return m_buf[idx];
                }
                const Point& operator()(ConstPointRef idx)
                {
                    return m_buf[idx];
                }
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
        struct Simplex: clarkson93::SimplexBase<ClarksonTraits>
        {
            Eigen::Matrix<double,2,1>   c_c;    ///< circum center
            double                      c_r;    ///< circum radius
            bool                        c_has;  ///< has center

            Simplex():c_r(0),c_has(false){}
        };

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
        template<typename T>
        struct Factory
        {
                T* m_start;    ///< start of our allocated buffer
                T* m_next;     ///< next unused block of memory
                T* m_end;      ///< past-the-end iterator

                /// initializes pointers to null
                Factory() :
                        m_start(0), m_end(0), m_next(0)
                {
                }

                /// frees any memory
                ~Factory()
                {
                    dealloc();
                }

                /// frees allocated memory if we have allocated any
                void dealloc()
                {
                    if (m_start)
                    {
                        ::operator delete(m_start);
                        m_start = 0;
                        m_end = 0;
                        m_next = 0;
                    }
                }

                /// allocates a block of memory large enough for n objects of type T
                void alloc(size_t n)
                {
                    dealloc();
                    if (n > 0)
                    {
                        m_start = (Simplex*) ::operator new(n * sizeof(T));
                        m_next = m_start;
                        m_end = m_start + n;
                    }
                }

                /// destroy all of the objects we have created and reset the memory
                /// pointer to the beginning of the buffer
                void clear()
                {
                    while (m_next > m_start)
                    {
                        --m_next;
                        m_next->~T();
                    }
                }

                /// construct an object of type T using the next available memory
                // slot and return it's pointer
                template<typename ...P>
                T* create(P ... params)
                {
                    assert(m_next != m_end);
                    T* next = m_next++;
                    return new (next) T(params...);
                }

                /// construct an object of type T using the next available memory
                // slot and return it's pointer
                T* create()
                {
                    assert(m_next != m_end);
                    T* next = m_next++;
                    return new (next) T();
                }
        };

        /// the triangulation provides some static callbacks for when hull faces
        /// are added or removed. If we wish to do anything special this is where
        /// we can hook into them. If you do not wish to hook into the callbacks
        /// then simply create an empty structure which has empty implementation
        /// for these
        struct Callback
        {
                void hullFaceAdded(Simplex* S)
                {
                }
                void hullFaceRemoved(Simplex* S)
                {
                }
        };

        /// the triangulation takes a reference to one of these objects in it's
        /// setup() method.
        struct Setup
        {
            Point* m_ptBuf;
            typedef unsigned int uint;

            Setup(Point* buf) :
                    m_ptBuf(buf)
            {
            }

            /// this is only required if NDim = Dynamic (-1), in which case
            /// this is how we tell the triangulation what the dimension is
            /// now
            uint nDim()
            {
                return NDim;
            }

            /// size of the hull set enumerator to preallocate
            uint hullPrealloc()
            {
                return 500;
            }

            /// size of the horizion set to preallocate
            uint horizonPrealloc()
            {
                return 500;
            }

            /// size of x_visible to preallocate
            uint xVisiblePrealloc()
            {
                return 1000;
            }

            /// returns a special PointRef which is used as the anti origin.
            PointRef antiOrigin()
            {
                return -1;
            }

            /// returns an object that can dereference a PointRef
            PointDeref deref()
            {
                return PointDeref(m_ptBuf);
            }

            /// returns a pointer to a callback object
            Callback callback()
            {
                return Callback();
            }

            /// sets up the allocator however we need it to
            void setupAlloc(Factory<Simplex>& alloc)
            {
                alloc.alloc(1000);
            }

        };

};

typedef ClarksonTraits::Simplex     Simplex;
typedef ClarksonTraits::Point       Point;
typedef ClarksonTraits::PointRef    PointRef;
typedef ClarksonTraits::PointDeref  PointDeref;

typedef clarkson93::Triangulation<ClarksonTraits>            Triangulation;
typedef clarkson93::Stack<Simplex*, clarkson93::SimplexBits> SimplexStack;
typedef brown79::Inversion<ClarksonTraits> Inversion;

/// calculates vertices of an arrow-head to draw at the end of a vector
/// from start to end
void calcVertexes(const Point& start, const Point& end, Point& p1, Point& p2,
        double length, double degrees)
{
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


void computeCenter( Inversion& inv, PointDeref& deref, Simplex* S )
{
    typedef Eigen::Matrix<double,2,2> Matrix;
    typedef Eigen::Matrix<double,2,1> Vector;

    // calculate circumcenter of the first simplex
    // see 2012-10-26-Note-11-41_circumcenter.xoj for math
    Matrix   A;
    Vector   b;

    Vector x0 = inv( deref(S->vertices[1]) ).block(0,0,2,1);
    for(int i=1; i < 3; i++)
    {
        Vector xi   = inv( deref(S->vertices[i+1]) ).block(0,0,2,1);
        Vector dv   = 2*(xi - x0);
        A.row(i-1)  = dv;
        b[i-1]      = xi.squaredNorm() - x0.squaredNorm();
    }

    // the circum center
    S->c_c  = A.fullPivLu().solve(b);

    // squared radius of the circumsphere
    S->c_r = ( S->c_c - x0 ).norm();
    S->c_has = true;
}

struct Main
{
        std::string m_layoutFile;
        std::string m_stateFile;

        gtk::LayoutMap m_layout;
        Gtk::Main m_gtkmm;
        View m_view;

        sigc::connection auto_cnx;

        Triangulation   m_T;
        Inversion       m_inv;
        SimplexStack    m_hull_Q;
        SimplexStack        m_hull_E;
        std::vector<Point>  m_ptStore;

        Main()
        {
            m_ptStore.reserve(1000);

            m_layoutFile = std::string(g_srcDir) + "/layout.glade";
            m_stateFile = std::string(g_binDir) + "/state.yaml";

            m_layout.loadLayout(m_layoutFile);
            m_layout.loadValues(m_stateFile);

            m_layout.widget<Gtk::AspectFrame>("viewFrame")->add(m_view);
            m_view.show();

            sigc::slot<void, const Cairo::RefPtr<Cairo::Context>&> slot_draw =
                    sigc::mem_fun(*this, &Main::draw);
            sigc::slot<void, double, double> slot_addPoint = sigc::mem_fun(
                    *this, &Main::addPoint);
            sigc::slot<void, double, double> slot_onMove = sigc::mem_fun(*this,
                    &Main::onMovement);

            m_view.sig_draw.connect(slot_draw);
            m_view.sig_mouse.connect(slot_onMove);
            m_view.sig_click.connect(slot_addPoint);

            m_layout.get<Gtk::Button>("btn_clear")->signal_clicked().connect(
                    sigc::mem_fun(this, &Main::clear));

            const char* keys[] = {
                    "chk_sites",     "chk_delaunay", "chk_balls",
                    "chk_vpoints",   "chk_edges" };

            for (int i = 0; i < 5; i++)
            {
                m_layout.get<Gtk::CheckButton>(keys[i])->signal_clicked().connect(
                        sigc::mem_fun(m_view, &View::queue_draw));
            }

            m_layout.get<Gtk::ToggleButton>("tbtn_auto")->signal_toggled().connect(
                    sigc::mem_fun(this, &Main::autoStateChanged));

            m_layout.get<Gtk::Button>("btn_random")->signal_clicked().connect(
                    sigc::mem_fun(this, &Main::random));
            m_layout.get<Gtk::Button>("btn_clear")->signal_clicked().connect(
                    sigc::mem_fun(this, &Main::clear));

            ClarksonTraits::Setup setup(m_ptStore.data());
            m_T.setup(setup);
            m_inv.init(Point(0.5, 0.5, 1), 1);

            m_hull_Q.setBit(clarkson93::simplex::ENUMERATE_QUEUED);
            m_hull_E.setBit(clarkson93::simplex::ENUMERATE_EXPANDED);

            m_hull_Q.reserve(1000);
            m_hull_E.reserve(1000);
        }

        ~Main()
        {
            m_layout.saveValues(m_stateFile);
        }

        void onMovement(double x, double y)
        {
            m_layout.get<Gtk::Adjustment>("adj_x")->set_value(x);
            m_layout.get<Gtk::Adjustment>("adj_y")->set_value(y);
        }

        void addPoint(double x, double y)
        {
            m_ptStore.push_back(m_inv(Point(x, y, 0)));

            if (m_ptStore.size() < 4)
            {
            }
            else if (m_ptStore.size() == 4)
                m_T.init(0, 4);
            else
            {
                PointRef idx = m_ptStore.size() - 1;
                m_T.insert(idx);
            }

            m_view.queue_draw();
        }

        void clear()
        {
            m_T.clear();
            m_ptStore.clear();
            m_view.queue_draw();
        }

        void run()
        {
            m_gtkmm.run(*m_layout.widget<Gtk::Window>("main"));
        }

        void random()
        {
            addPoint(rand() / (double) RAND_MAX, rand() / (double) RAND_MAX);
        }

        bool onSignal()
        {
            random();
            return true;
        }

        void autoStateChanged()
        {
            if (auto_cnx.empty())
            {
                auto_cnx = Glib::signal_timeout().connect(
                        sigc::mem_fun(*this, &Main::onSignal), 200);
            }
            else
                auto_cnx.disconnect();
        }

        void draw(const Cairo::RefPtr<Cairo::Context>& ctx)
        {
            gtk::EigenCairo ectx(ctx);
            ctx->set_line_width(0.001);

            if( m_layout.get<Gtk::CheckButton>("chk_sites")->get_active() )
            {
                Gdk::RGBA color =
                    m_layout.get<Gtk::ColorButton>("color_sites")->get_rgba();
                ectx.set_source_rgba(
                        color.get_red(),
                        color.get_green(),
                        color.get_blue(),
                        color.get_alpha() );
                for( const Point& p : m_ptStore )
                {
                    Point x0 = m_inv(p);
                    Eigen::Matrix<double,2,1> x1(x0[0],x0[1]);
                    ectx.circle(x1,0.01);
                    ectx.fill();
                }
            }

            if( m_ptStore.size() < 4)
                return;

            bool drawDelaunay =
                    m_layout.get<Gtk::CheckButton>("chk_delaunay")->get_active();
            bool drawCircumCenter =
                    m_layout.get<Gtk::CheckButton>("chk_vpoints")->get_active();
            bool drawCircumSphere =
                    m_layout.get<Gtk::CheckButton>("chk_balls")->get_active();
            bool drawVoronoi =
                    m_layout.get<Gtk::CheckButton>("chk_edges")->get_active();

            m_hull_Q.clear();
            m_hull_E.clear();

            // start the stack at some hull simplex
            m_hull_Q.push(m_T.m_hullSimplex);

            // pointer to the point buffer
            Point* pts = m_ptStore.data();

            PointDeref deref(m_ptStore.data());
            while (m_hull_Q.size() > 0)
            {
                // pop a simplex off the stack
                Simplex* S = m_hull_Q.pop();

                // it is a nearest site if the inversion center is not
                // visible, so draw it
                if (!S->isVisible(m_inv.center()))
                {
                    if( drawDelaunay )
                    {
                        Gdk::RGBA color =
                        m_layout.get<Gtk::ColorButton>("color_delaunay")->get_rgba();
                        ectx.set_source_rgba(
                                color.get_red(),
                                color.get_green(),
                                color.get_blue(),
                                color.get_alpha() );

                        for (int i = 0; i < 3; i++)
                        {
                            int j = (i + 1) % 3;
                            int ii = i + 1;
                            int jj = j + 1;
                            Point x0 = m_inv(pts[S->vertices[ii]]);
                            Point x1 = m_inv(pts[S->vertices[jj]]);
                            ectx.move_to((double) x0[0], (double) x0[1]);
                            ectx.line_to((double) x1[0], (double) x1[1]);
                        }

                        ectx.stroke();
                    }

                    computeCenter(m_inv,deref,S);

                    if( drawCircumCenter || drawCircumSphere )
                    {
                        if( drawCircumSphere )
                        {
                            Gdk::RGBA color =
                                m_layout.get<Gtk::ColorButton>("color_balls")->get_rgba();
                            ectx.set_source_rgba(
                                    color.get_red(),
                                    color.get_green(),
                                    color.get_blue(),
                                    color.get_alpha() );
                            ectx.circle(S->c_c,S->c_r);
                            ectx.fill();
                        }

                        if( drawCircumCenter )
                        {
                            Gdk::RGBA color =
                                m_layout.get<Gtk::ColorButton>("color_vpoints")->get_rgba();
                            ectx.set_source_rgba(
                                    color.get_red(),
                                    color.get_green(),
                                    color.get_blue(),
                                    color.get_alpha() );
                            ectx.circle(S->c_c,0.01);
                            ectx.fill();
                        }
                    }

                    if( drawVoronoi )
                    {
                        for(int i=1; i < 4; i++)
                        {
                            Simplex* N = S->neighbors[i];
                            if( !N->isVisible( m_inv.center() ) )
                            {
                                computeCenter(m_inv,deref,N);
                                Gdk::RGBA color =
                                m_layout.get<Gtk::ColorButton>("color_edges")->get_rgba();
                                ectx.set_source_rgba(
                                        color.get_red(),
                                        color.get_green(),
                                        color.get_blue(),
                                        color.get_alpha() );

                                ectx.move_to(S->c_c);
                                ectx.line_to(N->c_c);
                                ectx.stroke();
                            }
                        }
                    }
                }

                // in any case, expand it
                m_hull_E.push(S);

                // get all it's infinite neighbors
                for (int i = 1; i < 4; i++)
                {
                    Simplex* N = S->neighbors[i];
                    // if we haven't already queued or expanded it then queue
                    // it
                    if (!m_hull_Q.isMember(N) && !m_hull_E.isMember(N))
                        m_hull_Q.push(N);
                }
            }

        }
};

int main(int argc, char** argv)
{
    Main m_app;
    m_app.run();
    return 0;
}

