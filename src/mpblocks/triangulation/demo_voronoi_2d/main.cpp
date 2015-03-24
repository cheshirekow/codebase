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

using namespace mpblocks;


struct ClarksonTraits
{
    typedef unsigned int uint;
    static const unsigned int NDim = 3;
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, NDim, 1> Point;
    struct Simplex;
    typedef uint PointRef;
    typedef Simplex* SimplexRef;

    class Deref
    {
        private:
            std::vector<Point>* m_buf;

        public:
            Deref( std::vector<Point>* buf=0 ){ setBuf(buf); }
            void setBuf( std::vector<Point>* buf ) { m_buf = buf; }

            Point&      point(  PointRef idx ){ return (*m_buf)[idx]; }
            Simplex& simplex( SimplexRef ptr ){ return *ptr; }
    };

    struct Simplex:
            clarkson93::Simplex2<ClarksonTraits>
    {
        typedef clarkson93::Simplex2<ClarksonTraits> Base;

        Eigen::Matrix<double,2,1>   c_c;    ///< circum center
        double                      c_r;    ///< circum radius
        bool                        c_has;  ///< has center

        Simplex():
            Base(-1,0),
            c_r(0),
            c_has(false)
        {}
    };

    typedef clarkson93::SimplexOps< ClarksonTraits > SimplexOps;

    struct SimplexMgr:
        public std::vector<Simplex>
    {
        typedef std::vector<Simplex> base_t;

        SimplexRef create()
        {
            assert( base_t::size() < base_t::capacity() );
            base_t::emplace_back( );
            return &base_t::back();
        }
    };

    struct Callback
    {
        void hullFaceAdded(Simplex* S){}
        void hullFaceRemoved(Simplex* S){}
    };
};


typedef ClarksonTraits::Simplex     Simplex;
typedef ClarksonTraits::Point       Point;
typedef ClarksonTraits::PointRef    PointRef;
typedef ClarksonTraits::Deref       Deref;

typedef clarkson93::Triangulation<ClarksonTraits>            Triangulation;
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


void computeCenter( Inversion& inv, Deref& deref, Simplex& S )
{
    typedef Eigen::Matrix<double,2,2> Matrix;
    typedef Eigen::Matrix<double,2,1> Vector;

    // calculate circumcenter of the first simplex
    // see 2012-10-26-Note-11-41_circumcenter.xoj for math
    Matrix   A;
    Vector   b;

    PointRef peak = S.V[S.iPeak];
    std::vector<PointRef> pRefs;
    std::copy_if( S.V, S.V + 4,
                    std::back_inserter(pRefs),
                    [peak](PointRef v){ return peak != v; } );

    Vector x0 = inv( deref.point(pRefs[0]) ).block(0,0,2,1);
    for(int i=1; i < 3; i++)
    {
        Vector xi   = inv( deref.point(pRefs[i]) ).block(0,0,2,1);
        Vector dv   = 2*(xi - x0);
        A.row(i-1)  = dv;
        b[i-1]      = xi.squaredNorm() - x0.squaredNorm();
    }

    // the circum center
    S.c_c  = A.fullPivLu().solve(b);

    // squared radius of the circumsphere
    S.c_r = ( S.c_c - x0 ).norm();
    S.c_has = true;
}


struct Main
{
        std::string m_layoutFile;
        std::string m_stateFile;

        Gtk::Main m_gtkmm;
        gtk::LayoutMap  m_layout;
        gtk::SimpleView m_view;

        sigc::connection auto_cnx;

        Triangulation   m_T;
        Inversion       m_inv;
        std::vector<Point>  m_ptStore;

        Main()
        {
            m_layoutFile = std::string(g_srcDir) + "/layout.glade";
            m_stateFile = std::string(g_binDir) + "/state.yaml";

            m_layout.loadLayout(m_layoutFile);

            m_layout.widget<Gtk::AspectFrame>("viewFrame")->add(m_view);
            m_view.show();

            sigc::slot<void, const Cairo::RefPtr<Cairo::Context>&> slot_draw =
                    sigc::mem_fun(*this, &Main::draw);
            sigc::slot<void, double, double> slot_addPoint = sigc::mem_fun(
                    *this, &Main::addPoint);
            sigc::slot<void, double, double> slot_onMove = sigc::mem_fun(*this,
                    &Main::onMovement);

            m_view.sig_draw.connect(slot_draw);
            m_view.sig_motion.connect( slot_onMove );
            m_view.sig_press.connect(slot_addPoint);

            m_layout.get<Gtk::Button>("btn_clear")->signal_clicked().connect(
                    sigc::mem_fun(this, &Main::clear));


            std::vector<std::string> keys =
            {
                    "chk_sites",     "chk_delaunay", "chk_balls",
                    "chk_vpoints",   "chk_edges"
            };

            for ( const std::string& key : keys)
            {
                m_layout.get<Gtk::CheckButton>(key)->signal_clicked().connect(
                    sigc::mem_fun(m_view, &gtk::SimpleView::queue_draw));
            }

            m_layout.get<Gtk::ToggleButton>("tbtn_auto")->signal_toggled().connect(
                    sigc::mem_fun(this, &Main::autoStateChanged));

            m_layout.get<Gtk::Button>("btn_random")->signal_clicked().connect(
                    sigc::mem_fun(this, &Main::random));
            m_layout.get<Gtk::Button>("btn_clear")->signal_clicked().connect(
                    sigc::mem_fun(this, &Main::clear));
            m_layout.get<Gtk::Button>("saveImage")->signal_clicked().connect(
                    sigc::mem_fun(this, &Main::saveImage));

            m_layout.get<Gtk::Adjustment>("edgeWidth")
                    ->signal_value_changed().connect(
                            sigc::mem_fun(m_view,&gtk::SimpleView::queue_draw) );
            m_layout.get<Gtk::Adjustment>("pointRadius")
                    ->signal_value_changed().connect(
                            sigc::mem_fun(m_view,&gtk::SimpleView::queue_draw) );

            m_inv.init( Point(0.5,0.5,1), 1 );
            m_ptStore.reserve(1000);
            m_T.m_antiOrigin = -1;
            m_T.m_sMgr      .reserve(1000);
            m_T.m_ridges    .reserve( 1000 );
            m_T.m_deref.setBuf( &m_ptStore );
        }

        ~Main()
        {
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
                m_T.init(0, 4, [](uint i){return i;});
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
            m_layout.loadValues(m_stateFile);
            m_layout.widget<Gtk::Window>("main")->show_all_children(true);
            m_gtkmm.run(*m_layout.widget<Gtk::Window>("main"));
            m_layout.saveValues(m_stateFile);
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
            double eWidth = m_layout.get<Gtk::Adjustment>("edgeWidth")->get_value()/20000.0;
            double rPoint = m_layout.get<Gtk::Adjustment>("pointRadius")->get_value()/10000.0;


            gtk::EigenCairo ectx(ctx);
            ctx->set_line_width(eWidth);

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
                    ectx.circle(x1,rPoint);
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

            Deref  deref( &m_ptStore );
            Point* pts = m_ptStore.data();

            for( Simplex& S : m_T.m_sMgr )
            {
                if( !S.sets[ clarkson93::simplex::HULL ] )
                    continue;
                if( m_T.isVisible(S,m_inv.center()) )
                    continue;

                // it is a nearest site if the inversion center is not
                // visible, so draw it
                if( drawDelaunay )
                {
                    Gdk::RGBA color =
                    m_layout.get<Gtk::ColorButton>("color_delaunay")->get_rgba();
                    ectx.set_source_rgba(
                            color.get_red(),
                            color.get_green(),
                            color.get_blue(),
                            color.get_alpha() );

                    PointRef peak = S.V[S.iPeak];
                    std::vector<PointRef> pRefs;
                    std::copy_if( S.V, S.V + 4,
                                    std::back_inserter(pRefs),
                                    [peak](PointRef v){ return peak != v; } );

                    Point x0 = m_inv(pts[pRefs[2]]);
                    ectx.move_to((double) x0[0], (double) x0[1]);
                    for (int i = 0; i < 3; i++)
                    {
                        Point x1 = m_inv(pts[pRefs[i]]);
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
                        ectx.circle(S.c_c,S.c_r);
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
                        ectx.circle(S.c_c,rPoint);
                        ectx.fill();
                    }
                }

                if( drawVoronoi )
                {
                    for(int i=0; i < 4; i++)
                    {
                        Simplex* N = S.N[i];
                        if( N == m_T.peakNeighbor(S) )
                            continue;
                        if( !m_T.isVisible( *N, m_inv.center() ) )
                        {
                            computeCenter(m_inv,deref,*N);
                            Gdk::RGBA color =
                            m_layout.get<Gtk::ColorButton>("color_edges")->get_rgba();
                            ectx.set_source_rgba(
                                    color.get_red(),
                                    color.get_green(),
                                    color.get_blue(),
                                    color.get_alpha() );

                            ectx.move_to(S.c_c);
                            ectx.line_to(N->c_c);
                            ectx.stroke();
                        }
                    }
                }
            }
        }


        void saveImage()
        {
            Gtk::FileChooserDialog dialog("Save Image",
                    Gtk::FILE_CHOOSER_ACTION_SAVE);
            dialog.set_transient_for(*m_layout.get<Gtk::Window>("main"));

            //Add response buttons the the dialog:
            dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
            dialog.add_button(Gtk::Stock::OPEN, Gtk::RESPONSE_OK);

            //Add filters, so that only certain file types can be selected:

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

            //Show the dialog and wait for a user response:
            int result = dialog.run();

            //Handle the response:
            switch(result)
            {
                case(Gtk::RESPONSE_OK):
                {
                    std::string filename = dialog.get_filename();
                    int n = filename.size();
                    if( n < 5 )
                    {
                        std::cout << "invalid filename: " << filename << std::endl;
                        return;
                    }

                    std::string ext = filename.substr(n-4,4);
                    if( ext.compare(".png") == 0)
                    {
                        int w = 800;
                        int h = 800;

                        Cairo::RefPtr<Cairo::ImageSurface> img =
                            Cairo::ImageSurface::create(Cairo::FORMAT_ARGB32,w,h);
                        Cairo::RefPtr<Cairo::Context> ctx =
                                Cairo::Context::create(img);

                        ctx->scale(w,-h);
                        ctx->translate(0,-1);
                        ctx->set_source_rgb(1,1,1);
                        ctx->paint();

                        draw(ctx);

                        img->write_to_png(filename);
                    }
                    else if( ext.compare(".pdf") == 0)
                    {
                        int w = 400;
                        int h = 400;
                        Cairo::RefPtr<Cairo::PdfSurface> img =
                            Cairo::PdfSurface::create(filename,w,h);
                        Cairo::RefPtr<Cairo::Context> ctx =
                                Cairo::Context::create(img);

                        ctx->scale(w,-h);
                        ctx->translate(0,-1);
                        ctx->set_source_rgb(1,1,1);
                        ctx->paint();

                        draw(ctx);
                    }
                    else if( ext.compare(".svg") == 0)
                    {
                        int w = 400;
                        int h = 400;
                        Cairo::RefPtr<Cairo::SvgSurface> img =
                            Cairo::SvgSurface::create(filename,w,h);
                        Cairo::RefPtr<Cairo::Context> ctx =
                                Cairo::Context::create(img);

                        ctx->scale(w,-h);
                        ctx->translate(0,-1);
                        ctx->set_source_rgb(1,1,1);
                        ctx->paint();

                        draw(ctx);
                    }
                    else
                    {
                        std::cout << "invalid file format: " << filename << std::endl;
                        return;
                    }

                    break;
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

