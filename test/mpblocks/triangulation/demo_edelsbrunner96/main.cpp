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

#include <mpblocks/edelsbrunner96.hpp>
#include <mpblocks/gtk.hpp>

#include <cstdlib>
#include <gtkmm.h>
#include <iostream>
#include <string>




using namespace mpblocks;
using namespace mpblocks::edelsbrunner96;

typedef Eigen::Matrix<double,2,1> Vector;
typedef Eigen::Matrix<double,2,1> Point;

typedef ExampleTraits             Traits;
typedef Traits::Simplex           Simplex;
typedef Triangulation<Traits>     Triangulation_t;





struct Main
{
    Gtk::Main           m_gtkmm;

    std::vector<Point>  m_ptStore;
    gtk::SimpleView     m_view;
    gtk::LayoutMap      m_layout;
    Triangulation_t     m_T;

    std::string     m_gladeFile;
    std::string     m_stateFile;

    sigc::connection    auto_cnx;

    Main()
    {
        m_gladeFile = std::string(g_srcDir) + "/ui.glade";
        m_stateFile = std::string(g_binDir) + "/demo_clarkson93.yaml";

        m_ptStore.reserve(1000);

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

        const char* chk_list[] = {
            0
        };

        for( const char** chk = chk_list; *chk; ++chk )
        {
            m_layout.get<Gtk::CheckButton>(*chk)
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
            m_T.init( &m_ptStore[0], &m_ptStore[3], [](Point* ptr){return ptr;} );
        else
            m_T.insert(&m_ptStore.back());

        m_view.queue_draw();
    }

    void onClick( double x, double y )
    {
        addPoint(x,y);
    }

    void clear()
    {
        m_T.clear();
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






