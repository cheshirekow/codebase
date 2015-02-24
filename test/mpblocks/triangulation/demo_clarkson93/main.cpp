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



#include "config.h"

#include <mpblocks/clarkson93.hpp>
#include <mpblocks/gtk.hpp>

#include <cstdlib>
#include <gtkmm.h>
#include <iostream>
#include <string>




using namespace mpblocks;
using namespace mpblocks::clarkson93;

typedef Eigen::Matrix<double,2,1> Vector;
typedef Eigen::Matrix<double,2,1> Point;

typedef ExampleTraits2<double,2>  Traits;
typedef typename Traits::Simplex  Simplex;
typedef Triangulation<Traits>     Triangulation_t;


void calcVertexes(const Point& start, const Point& end,
                    Point& p1, Point& p2,
                    double length,
                    double degrees )
{
    double end_y    = end[1];
    double start_y  = start[1];
    double end_x    = end[0];
    double start_x  = start[0];

    double angle = atan2 (end_y - start_y, end_x - start_x) + M_PI;

    p1[0] = end_x + length * cos(angle - degrees);
    p1[1] = end_y + length * sin(angle - degrees);
    p2[0] = end_x + length * cos(angle + degrees);
    p2[1] = end_y + length * sin(angle + degrees);
}



struct Main
{
    Gtk::Main           m_gtkmm;

    std::vector<Point>  m_ptStore;
    gtk::SimpleView     m_view;
    gtk::LayoutMap      m_layout;
    Triangulation_t     m_hull;

    std::string     m_gladeFile;
    std::string     m_stateFile;

    sigc::connection    auto_cnx;

    Main()
    {
        m_gladeFile = std::string(g_srcDir) + "/ui.glade";
        m_stateFile = std::string(g_binDir) + "/demo_clarkson93.yaml";

        m_ptStore.reserve(10000);
        m_hull.m_antiOrigin = 0;
        m_hull.m_sMgr.reserve(10000);


        m_layout.loadLayout(m_gladeFile);

        m_layout.get<Gtk::AspectFrame>("aspect_view")->add(m_view);
        m_view.show();


        m_view.sig_motion.connect(
                sigc::mem_fun(*this,&Main::onMovement) );
        m_view.sig_press.connect(
                sigc::mem_fun(*this,&Main::onClick) );
        m_view.sig_draw.connect(
                sigc::mem_fun(*this,&Main::draw) );

        m_layout.get<Gtk::Button>("btn_clear")->signal_clicked()
            .connect( sigc::mem_fun(*this,&Main::clear) );

        m_layout.get<Gtk::ToggleButton>("tbtn_auto")->signal_toggled()
                .connect( sigc::mem_fun(*this, &Main::autoStateChanged) );

        m_layout.get<Gtk::Button>("btn_random")->signal_clicked()
            .connect( sigc::mem_fun(*this, &Main::random) );

        std::vector<std::string> chk_list =
        {
            "chk_triangulation",
            "chk_neighbors",
            "chk_xvh",
            "chk_xv_walked",
            "chk_hull",
            "chk_ridges",
            "chk_hfill",
            "chk_xvh_flag",
            "chk_xv_walked_flag",
            "chk_ridges_flag",
            "chk_hfill_flag",
        };

        for( const std::string& key : chk_list )
        {
            m_layout.get<Gtk::CheckButton>(key)
                ->signal_clicked().connect( sigc::mem_fun(
                        m_view,&gtk::SimpleView::queue_draw) );
        }
    }


    void onMovement( double x, double y )
    {
        m_layout.get<Gtk::Adjustment>("adj_x")->set_value(x);
        m_layout.get<Gtk::Adjustment>("adj_y")->set_value(y);
    }

    void addPoint(double x, double y)
    {
        m_ptStore.emplace_back(x,y);
        if( m_ptStore.size() < 3 )
            return;
        else if( m_ptStore.size() == 3)
            m_hull.init( &m_ptStore[0], &m_ptStore[3], [](Point* ptr){return ptr;} );
        else
            m_hull.insert(&m_ptStore.back());

        m_view.queue_draw();
    }

    void onClick( double x, double y )
    {
        addPoint(x,y);
    }

    void clear()
    {
        m_hull.m_xv_walked.clear();
        m_hull.m_xvh.clear();
        m_hull.m_ridges.clear();

        m_hull.clear();
        m_ptStore.clear();
        m_view.queue_draw();
    }

    void random()
    {
        addPoint(  rand() / (double)RAND_MAX,
                   rand() / (double)RAND_MAX  );
    }

    bool autoStep()
    {
        random();
        return true;
    }

    void autoStateChanged(  )
    {
        auto_cnx.disconnect();
        if( m_layout.get<Gtk::ToggleButton>("tbtn_auto")->get_active() )
        {
            auto_cnx = Glib::signal_timeout().connect(
                    sigc::mem_fun(*this,&Main::autoStep), 50 );
        }

    }

    void draw( const Cairo::RefPtr<Cairo::Context>& ctx )
    {
        if(m_ptStore.size() < 3)
            return;

        gtk::EigenCairo ectx(ctx);

        double r = 0.005;   // circle radius for points
        double w = 0.001;   // line width
        double ns = 0.05;   // size of drawn normals
        double as = 0.01;   // length of arrow heads

        Cairo::RefPtr<Cairo::SolidPattern> triangulationColor =
                Cairo::SolidPattern::create_rgb( 0.7, 0.9, 0.7 );

        Cairo::RefPtr<Cairo::SolidPattern> constraintColor =
                Cairo::SolidPattern::create_rgb( 0.7, 0.9, 0.7 );

        Cairo::RefPtr<Cairo::SolidPattern> xVisibleColor =
                Cairo::SolidPattern::create_rgb( 0.4, 0.9, 0.4 );

        Cairo::RefPtr<Cairo::SolidPattern> hullColor =
                Cairo::SolidPattern::create_rgb( 0.7, 0.7, 0.9 );

        Cairo::RefPtr<Cairo::SolidPattern> vertexColor =
                Cairo::SolidPattern::create_rgb( 0.9, 0.7, 0.7 );

        Cairo::RefPtr<Cairo::SolidPattern> neighborColor =
                Cairo::SolidPattern::create_rgb( 0.6, 0.6, 0.6 );

        Cairo::RefPtr<Cairo::SolidPattern> ridgeColor =
                Cairo::SolidPattern::create_rgb( 0.6, 0.9, 0.6 );



        // draw the triangulation
        if( m_layout.get<Gtk::CheckButton>("chk_triangulation")->get_active() )
        {
            for( Simplex& S : m_hull.m_sMgr )
            {
                if( S.V[S.iPeak] == m_hull.m_antiOrigin )
                    continue;

                ectx.move_to((*S.V[2]));
                for(int i=0; i < 3; i++)
                    ectx.line_to( (*S.V[i]) );

                ctx->set_source(triangulationColor);
                ctx->set_line_width(w);
                ctx->stroke();

                // don't draw the base vertex normal if this is a boundary simplex
                // if( (*ipSimplex)->vertices[0] == &convex_hull.antiOrigin )
                //    continue;

                // average the two base points
//                Point a = (*(p[1]) + *(p[2]) ) / 2.0;
//                Point b = a - ns * S.n;
//                Point v1, v2;
//
//                calcVertexes(a,b,v1,v2,as,M_PI/8);
//
//                ectx.move_to(a);
//                ectx.line_to(b);
//                ctx->stroke();
//
//                ectx.move_to(b);
//                ectx.line_to(v2);
//                ectx.line_to(v1);
//                ectx.line_to(b);
//                ctx->fill();
            }
        }

        if( m_layout.get<Gtk::CheckButton>("chk_neighbors")->get_active() )
        {
            for( Simplex& S : m_hull.m_sMgr )
            {
                Point p_a, p_b, p_c, v1,v2;

                if( S.V[S.iPeak] == m_hull.m_antiOrigin )
                {
                    uint32_t i=0;
                    uint32_t j=1;
                    if( i == S.iPeak )
                        i = 2;
                    if( j == S.iPeak )
                        j = 2;

                    p_a = ( *(S.V[i]) + *(S.V[j]) )/2.0 - ns*S.n;
                }
                else
                {
                    p_a = (1/3.0)*(*(S.V[0]))
                            + (1/3.0)*(*(S.V[1]))
                            + (1/3.0)*(*(S.V[2]));
                }

                for(int i=0; i < 3; i++)
                {
                    Simplex* N = S.N[i];

                    if( N->V[N->iPeak] == m_hull.m_antiOrigin )
                    {
                        uint32_t i=0;
                        uint32_t j=1;
                        if( i == N->iPeak )
                            i = 2;
                        if( j == N->iPeak )
                            j = 2;
                        p_b = ( *(N->V[i]) + *(N->V[j]) )/2.0 - ns*N->n;
                    }
                    else
                    {
                        p_b = (1/3.0)*(*(N->V[0]))
                            + (1/3.0)*(*(N->V[1]))
                            + (1/3.0)*(*(N->V[2]));
                    }

                    ectx.move_to(p_a);
                    ectx.line_to(p_b);
                    ctx->set_source( neighborColor );
                    ctx->set_line_width(w);
                    ctx->stroke();

                    // arrow head
                    p_c = p_a + (2.0/3.0)*(p_b - p_a);
                    calcVertexes(p_a,p_c,v1,v2,as,M_PI/8);

                    ectx.move_to( p_c);
                    ectx.line_to( v1 );
                    ectx.line_to( v2 );
                    ectx.line_to( p_c );
                    ctx->fill();
                }
            }
        }

        if( m_layout.get<Gtk::CheckButton>("chk_xvh")->get_active() )
        {
            for( Simplex* S : m_hull.m_xvh )
            {
                Point* peak = S->V[S->iPeak];
                std::vector<Point*> p;
                std::copy_if( S->V, S->V+3,
                                std::back_inserter(p),
                                [peak](Point* v){ return v != peak; } );

                Point     p1 = *(p[0]) - (r/2)*S->n;
                Point     p2 = *(p[1]) - (r/2)*S->n;

                ectx.move_to(p1);
                ectx.line_to(p2);
                ctx->set_source(xVisibleColor);
                ctx->set_line_width(2*w);
                ctx->stroke();
            }
        }

        if( m_layout.get<Gtk::CheckButton>("chk_xvh_flag")->get_active() )
        {
            for( Simplex& S : m_hull.m_sMgr )
            {
                if( !S.sets[ clarkson93::simplex::XV_HULL] )
                    continue;

                Point* peak = S.V[S.iPeak];
                std::vector<Point*> p;
                std::copy_if( S.V, S.V+3,
                                std::back_inserter(p),
                                [peak](Point* v){ return v != peak; } );

                Point     p1 = *(p[0]) - (r/2)*S.n;
                Point     p2 = *(p[1]) - (r/2)*S.n;

                ectx.move_to(p1);
                ectx.line_to(p2);
                ctx->set_source(hullColor);
                ctx->set_line_width(2*w);
                ctx->stroke();
            }
        }

        if( m_layout.get<Gtk::CheckButton>("chk_xv_walked")->get_active() )
        {
            for( Simplex* S : m_hull.m_xv_walked )
            {
                Point* peak = S->V[S->iPeak];
                std::vector<Point*> p;
                std::copy_if( S->V, S->V+3,
                                std::back_inserter(p),
                                [peak](Point* v){ return v != peak; } );

                Point     p1 = *(p[0]) - (r/2)*S->n;
                Point     p2 = *(p[1]) - (r/2)*S->n;

                ectx.move_to(p1);
                ectx.line_to(p2);
                ctx->set_source(xVisibleColor);
                ctx->set_line_width(2*w);
                ctx->stroke();
            }
        }

        if( m_layout.get<Gtk::CheckButton>("chk_xv_walked_flag")->get_active() )
        {
            for( Simplex& S : m_hull.m_sMgr )
            {
                if( !S.sets[ clarkson93::simplex::XVISIBLE_WALK] )
                    continue;

                Point* peak = S.V[S.iPeak];
                std::vector<Point*> p;
                std::copy_if( S.V, S.V+3,
                                std::back_inserter(p),
                                [peak](Point* v){ return v != peak; } );

                Point     p1 = *(p[0]) - (r/2)*S.n;
                Point     p2 = *(p[1]) - (r/2)*S.n;

                ectx.move_to(p1);
                ectx.line_to(p2);
                ctx->set_source(hullColor);
                ctx->set_line_width(2*w);
                ctx->stroke();
            }
        }

        if( m_layout.get<Gtk::CheckButton>("chk_hull")->get_active() )
        {
            for( Simplex& S : m_hull.m_sMgr )
            {
                if( !S.sets[ clarkson93::simplex::HULL] )
                    continue;

                Point* peak = S.V[S.iPeak];
                std::vector<Point*> p;
                std::copy_if( S.V, S.V+3,
                                std::back_inserter(p),
                                [peak](Point* v){ return v != peak; } );

                Point     p1 = *(p[0]) - (r/2)*S.n;
                Point     p2 = *(p[1]) - (r/2)*S.n;

                ectx.move_to(p1);
                ectx.line_to(p2);
                ctx->set_source(hullColor);
                ctx->set_line_width(2*w);
                ctx->stroke();
            }
        }

        bool drawRidges  = m_layout.get<Gtk::CheckButton>("chk_ridges")->get_active();
        bool drawFillers = m_layout.get<Gtk::CheckButton>("chk_hfill")->get_active();
        if( drawRidges || drawFillers )
        {
            for( auto& pair : m_hull.m_ridges )
            {
                Simplex& V = *pair.Svis;
                Simplex& S = *pair.Sinvis;
                std::vector<Point*> ridge;
                m_hull.vsetIntersection(V,S, std::back_inserter(ridge) );

                if( drawRidges )
                {
                    Point* peak = S.V[S.iPeak];
                    std::vector<Point*> p;
                    std::copy_if( S.V, S.V+3,
                                    std::back_inserter(p),
                                    [peak](Point* v){ return v != peak; } );

                    Point     p1 = *(p[0]) - (r/2)*S.n;
                    Point     p2 = *(p[1]) - (r/2)*S.n;

                    ectx.move_to(p1);
                    ectx.line_to(p2);
                    ctx->set_source(hullColor);
                    ctx->set_line_width(2*w);
                    ctx->stroke();

                    ectx.circle(*ridge[0],2*r);
                    ctx->set_source(ridgeColor);
                    ctx->fill();
                }
            }
        }


        // draw the vertices on the very top
        for( Point& p : m_ptStore )
        {
            ectx.circle(p,r);
            ctx->set_source(vertexColor);
            ctx->fill();
        }
    }

    void run()
    {
        m_layout.loadValues(m_stateFile);
        m_gtkmm.run( *m_layout.get<Gtk::Window>("main") );
        m_layout.saveValues(m_stateFile);
    }


};



int main(int argc, char** argv)
{
    Main app;
    app.run();
    return 1;
}






