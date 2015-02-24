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
 *  \file   test/demo/main.cpp
 *
 *  \date   Oct 25, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include "config.h"

#include <iostream>
#include <cstdlib>
#include <set>
#include <Eigen/Dense>
#include <gtkmm.h>
#include <mpblocks/gtk.hpp>



namespace mpblocks             {
namespace triangulation        {
namespace demo_uniform_simplex {


typedef Eigen::Matrix<float,2,1> Point;

class Main
{
    private:
        std::string m_layoutFile;
        std::string m_stateFile;

        Gtk::Main        m_gtkmm;
        gtk::LayoutMap   m_layout;
        gtk::SimpleView  m_view;
        sigc::connection m_genCnx;


        std::vector<Point>  m_ptStore;
        std::vector<Point>  m_simplex;

        Point m_mouse;

    public:
        Main(int argc, char** argv):
            m_gtkmm(argc,argv),
            m_mouse(0,0)
        {
            m_ptStore.reserve(1000);

            m_layoutFile = std::string(g_srcDir) +  "/layout.glade";
            m_stateFile  = std::string(g_binDir) +  "/state.yaml";

            m_layout.loadLayout( m_layoutFile );
            m_layout.loadValues( m_stateFile );

            m_layout.get<Gtk::AspectFrame>("viewFrame")->add(m_view);

            m_view.show();

            m_view.sig_draw  .connect( sigc::mem_fun(*this,&Main::draw) );
            m_view.sig_motion.connect( sigc::mem_fun(*this,&Main::onMove) );
            m_view.sig_press .connect( sigc::mem_fun(*this,&Main::addVertex) );

            m_layout.get<Gtk::Button>("drawSimplex")->signal_clicked()
                    .connect( sigc::mem_fun(*this,&Main::startSimplex));
            m_layout.get<Gtk::Button>("saveImage")->signal_clicked()
                    .connect( sigc::mem_fun(*this,&Main::saveImage));
            m_layout.get<Gtk::ToggleButton>("generateSamples")->signal_clicked()
                    .connect( sigc::mem_fun(*this,&Main::toggleGenerate));

            m_layout.get<Gtk::ColorButton>("pointColor")->signal_color_set()
                    .connect( sigc::mem_fun(m_view,&gtk::SimpleView::queue_draw) );
            m_layout.get<Gtk::ColorButton>("lineColor")->signal_color_set()
                    .connect( sigc::mem_fun(m_view,&gtk::SimpleView::queue_draw) );

        }

        ~Main()
        {
            m_layout.saveValues( m_stateFile );
        }

        void startSimplex(  )
        {
            m_simplex.clear();
            m_ptStore.clear();
        }

        void onMove( double x, double y )
        {
            m_mouse << x,y;
            if(m_simplex.size() < 3 )
                m_view.queue_draw();
        }

        void addVertex( double x, double y )
        {
            if(m_simplex.size() < 3 )
            {
                m_simplex.push_back( Point(x,y) );
                m_view.queue_draw();
            }
        }

        bool addPoints()
        {
            for(int i=0; i < 500; i++)
                addPoint();
            return true;
        }

        void addPoint(  )
        {
            if(m_simplex.size() < 3)
            {
                m_genCnx.disconnect();
                return;
            }

            std::set<double> z;
            for(int i=0; i < 2; i++)
                z.insert( rand() / (double)RAND_MAX );
            z.insert(1);

            double lambda[3];
            double z_prev=0;
            int    i=0;
            for( double z_i : z )
            {
                lambda[i++] = z_i - z_prev;
                z_prev = z_i;
            }

            Point x = lambda[0] * m_simplex[0]
                    + lambda[1] * m_simplex[1]
                    + lambda[2] * m_simplex[2];

            m_ptStore.push_back( x );
            m_view.queue_draw();

        }

        void toggleGenerate()
        {
            if( m_layout.get<Gtk::ToggleButton>("generateSamples")->get_active() )
            {
                m_genCnx.disconnect();
                m_genCnx = Glib::signal_timeout().connect(
                        sigc::mem_fun( *this, &Main::addPoints ), 100 );
            }
            else
            {
                m_genCnx.disconnect();
            }
        }

        void run()
        {
            m_gtkmm.run( *m_layout.widget<Gtk::Window>("main") );
        }

        void draw( const Cairo::RefPtr<Cairo::Context>& ctx )
        {
            gtk::EigenCairo ectx(ctx);

            if( m_simplex.size() < 1 )
                return;

            ectx.move_to( m_simplex[0] );

            for(int i=1; i < 3; i++)
            {
                if( m_simplex.size() < i+1 )
                    ectx.line_to( m_mouse );
                else
                    ectx.line_to( m_simplex[i] );
            }

            if( m_simplex.size() >= 3 )
                ectx.line_to( m_simplex[0] );

            Gdk::RGBA color =
                    m_layout.get<Gtk::ColorButton>("lineColor")->get_rgba();
            ectx.set_line_width(0.002);
            ectx.set_source_rgba(
                    color.get_red(),
                    color.get_green(),
                    color.get_blue(),
                    color.get_alpha());
            ectx.stroke();

            ectx.set_source_rgba(0,0,1,0.3);

            for( const Point& x : m_ptStore )
                ectx.circle( x, 0.001 );

            color = m_layout.get<Gtk::ColorButton>("pointColor")->get_rgba();
            ectx.set_source_rgba(
                    color.get_red(),
                    color.get_green(),
                    color.get_blue(),
                    color.get_alpha());
            ectx.fill();
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
                        int w = 100;
                        int h = 100;
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
                        int w = 100;
                        int h = 100;
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



} // demo_uniform_simplex
} // triangulation
} // mpblocks



namespace ns = mpblocks::triangulation::demo_uniform_simplex;


int main(int argc, char** argv)
{
    ns::Main m_app(argc,argv);
    m_app.run();
    return 0;
}








