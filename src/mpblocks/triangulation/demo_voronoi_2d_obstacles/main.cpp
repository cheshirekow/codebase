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
#include "ViewArea.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <loki/Typelist.h>
#include <mpblocks/voronoi_diagram/brown.hpp>
#include <mpblocks/collision/aabb_tree.hpp>

#include <cstdlib>
#include <gtkmm.h>
#include <iostream>
#include <string>
#include <fstream>


using namespace mpblocks::voronoi_diagram::brown;
using namespace mpblocks::voronoi_diagram::two_d_obstacle_set_creator;
using namespace mpblocks::collision;

typedef double                    Format_t;
typedef Diagram<Format_t,2>       Diagram_t;
//typedef Iterator<double,2>        Iterator_t;
typedef Diagram_t::LowVector_t    Vector_t;
typedef Diagram_t::LowPoint_t     Point_t;

typedef boost::mt19937                  Generator_t;
typedef boost::uniform_real<Format_t>   Distribution_t;

struct PolytopeTraits
{
    typedef ::Format_t Format_t;
    static const unsigned int NDim = 2;
};

typedef geometry::Polytope< PolytopeTraits >    Polytope_t;











struct Main
{
    Diagram_t       diagram;
    Generator_t     m_gen;
    Distribution_t  m_distx;
    Distribution_t  m_disty;
    bool            m_needsDraw;

    const Diagram_t::Cell_t*            m_activeCell;
    std::set<const Diagram_t::Cell_t* > m_selectedCells;

    ViewArea                        view;
    Glib::RefPtr<Gtk::Adjustment>   adj_generator_sites;

    Glib::RefPtr<Gtk::Adjustment>   adj_ws_x;
    Glib::RefPtr<Gtk::Adjustment>   adj_ws_y;
    Glib::RefPtr<Gtk::Adjustment>   adj_ws_w;
    Glib::RefPtr<Gtk::Adjustment>   adj_ws_h;

    Cairo::RefPtr<Cairo::SolidPattern> siteColor;
    Cairo::RefPtr<Cairo::SolidPattern> edgeColor;
    Cairo::RefPtr<Cairo::SolidPattern> activeColor;
    Cairo::RefPtr<Cairo::SolidPattern> selectedColor;

    Gtk::Window*        win_main;
    Gtk::CheckButton*   chk_draw_active;
    Gtk::Button*        btn_save;

    Gtk::RadioButton*   radio_add;
    Gtk::RadioButton*   radio_remove;

    bool onMouseMotion( GdkEventMotion* event )
    {
        // if the current active cell is cells.end() then we need to do the
        // initial search for the active cell
        if( !m_activeCell )
            return true;

        // convert mouse coordinates into workspace coordinates
        double x =  adj_ws_x->get_value() +
                    ( event->x / view.get_allocated_width() )
                        * adj_ws_w->get_value();
        double y = adj_ws_y->get_value() +
                    ( 1.0 - ( event->y / view.get_allocated_height() ) )
                        * adj_ws_h->get_value();

        // now start at the current active cell, and then move to the
        // neighboring cell which is nearest to the mouse pointer, until
        // there is no nearer neighbor
        Diagram_t::Point_t p( x, y, 0 );

        const Diagram_t::Cell_t*    thisCell = m_activeCell;
        const Diagram_t::Cell_t*    nextCell = m_activeCell;

        double minDist2 = (p - *(nextCell->site)).squaredNorm();
        double dist2    = 0;

        Diagram_t::Cell_t::FaceMap_t::const_iterator iFace;
        while( nextCell )
        {
            thisCell = nextCell;
            nextCell = 0;

            for( iFace = thisCell->faces.begin();
                    iFace != thisCell->faces.end();
                    ++iFace )
            {
                const Diagram_t::Point_t* pP = iFace->first;
                double dist2 = (p - *pP).squaredNorm();
                if( dist2 < minDist2 )
                {
                    minDist2   = dist2;
                    nextCell   = &( diagram.cells().find( pP )->second );
                }
            }
        }

        m_activeCell = thisCell;

        if(event->state & Gdk::BUTTON1_MASK )
        {
            if(radio_add->get_active())
                m_selectedCells.insert(m_activeCell);
            else
                m_selectedCells.erase(m_activeCell);
            m_needsDraw = true;
        }

        //view.queue_draw();
        return true;
    }

    void loadLayout()
    {
        Glib::RefPtr<Gtk::Builder> builder =
            Gtk::Builder::create_from_file(g_srcDir + "/ui.glade");

        builder->get_widget("window",win_main);

        Gtk::AspectFrame* frame;
        builder->get_widget("aspect_view",frame);

        adj_generator_sites =
                Glib::RefPtr<Gtk::Adjustment>::cast_static(
                    builder->get_object("adj_generator_sites") );

        adj_ws_x = Glib::RefPtr<Gtk::Adjustment>::cast_static(
                        builder->get_object("adj_ws_x") );
        adj_ws_y = Glib::RefPtr<Gtk::Adjustment>::cast_static(
                        builder->get_object("adj_ws_y") );
        adj_ws_w = Glib::RefPtr<Gtk::Adjustment>::cast_static(
                        builder->get_object("adj_ws_w") );
        adj_ws_h = Glib::RefPtr<Gtk::Adjustment>::cast_static(
                        builder->get_object("adj_ws_h") );

        adj_ws_x->signal_value_changed().connect(
                sigc::mem_fun(*this,&Main::sizeChanged) );
        adj_ws_y->signal_value_changed().connect(
                sigc::mem_fun(*this,&Main::sizeChanged) );
        adj_ws_w->signal_value_changed().connect(
                sigc::mem_fun(*this,&Main::sizeChanged) );
        adj_ws_h->signal_value_changed().connect(
                sigc::mem_fun(*this,&Main::sizeChanged) );

        view.sig_draw
                .connect( sigc::mem_fun(*this,&Main::onDraw) );

        Gtk::Button* btn;
        builder->get_widget("btn_generate",btn);
        btn->signal_clicked().connect(
                sigc::mem_fun(*this,&Main::generateCells) );
        builder->get_widget("btn_saveImage",btn);
        btn->signal_clicked().connect(
                sigc::mem_fun(*this,&Main::saveImage) );

        builder->get_widget("btn_save",btn_save);
        builder->get_widget("chk_draw_active",chk_draw_active);
        builder->get_widget("radio_add",radio_add);
        builder->get_widget("radio_remove",radio_remove);

        btn_save->signal_clicked().connect(
                sigc::mem_fun(*this,&Main::onSave) );

        frame->add( view );
        win_main->show_all();
    }

    void generateCells()
    {
        int numSites = adj_generator_sites->get_value();
        std::cout << "Generating " << numSites << " sites" << std::endl;

        diagram.clear();
        m_selectedCells.clear();
        m_distx = Distribution_t(0,adj_ws_w->get_value());
        m_disty = Distribution_t(0,adj_ws_h->get_value());
        for(int i=0; i < numSites; i++)
        {
            Point_t p;
            p << adj_ws_x->get_value() + m_distx(m_gen),
                 adj_ws_y->get_value() + m_disty(m_gen);
            diagram.insert(p);
        }

        diagram.buildCells();
        m_activeCell = &( (diagram.cells().begin())->second );
        //view.queue_draw();

        m_needsDraw = true;
    }


    void drawCell( const Cairo::RefPtr<Cairo::Context>& ctx,
                    const Diagram_t::Cell_t& cell )
    {
        Diagram_t::Cell_t::FaceMap_t::const_iterator iFace;
        for( iFace = cell.faces.begin();
                iFace != cell.faces.end();
                iFace++ )
        {
            const Diagram_t::Cell_t::PointSet_t& face = iFace->second;
            if( face.size() > 1 )
            {
                Diagram_t::Cell_t::PointSet_t::const_iterator
                    ipPoint = face.begin();

                ctx->move_to( (double) (**ipPoint)[0],
                              (double) (**ipPoint)[1]);
                ipPoint++;
                ctx->line_to( (double) (**ipPoint)[0],
                              (double) (**ipPoint)[1]);
                ctx->stroke();
            }
        }
    }

    void fillCell( const Cairo::RefPtr<Cairo::Context>& ctx,
                    const Diagram_t::Cell_t& cell )
    {
        typedef std::map< double, const Diagram_t::Point_t* > Ordered_t;

        Ordered_t vertices;
        Diagram_t::Cell_t::PointSet_t::const_iterator ipPoint;
        for( ipPoint = cell.vertices.begin();
                ipPoint != cell.vertices.end();
                ++ipPoint )
        {
            Diagram_t::Vector_t v =  ((**ipPoint) - *(cell.site) );
            double a = std::atan2( (double) v[1], (double) v[0] );
            vertices[a] = *ipPoint;
        }

        Ordered_t::iterator iPair = vertices.end();
        --iPair;

        ctx->move_to( (double) (*(iPair->second))[0],
                      (double) (*(iPair->second))[1] );

        for( iPair = vertices.begin();
                iPair != vertices.end();
                ++iPair )
        {
            ctx->line_to( (double) (*(iPair->second))[0],
                          (double) (*(iPair->second))[1] );
        }

        ctx->fill();
    }


    void sizeChanged()
    {
        view.setArea(
                adj_ws_x->get_value(),
                adj_ws_y->get_value(),
                adj_ws_w->get_value(),
                adj_ws_h->get_value() );
    }

    Main()
    {
        m_needsDraw = true;
        loadLayout();
        diagram.inversion().init( Diagram_t::Point_t(0.5,0.5,1), 1 );
        m_activeCell = 0;

        view.signal_motion_notify_event().connect(
                sigc::mem_fun(*this,&Main::onMouseMotion) );

        siteColor =
                Cairo::SolidPattern::create_rgb( 0.7, 0.9, 0.7 );

        edgeColor =
                Cairo::SolidPattern::create_rgb( 0.7, 0.9, 0.7 );

        activeColor =
                Cairo::SolidPattern::create_rgb( 1.0, 0.5, 0.0 );

        selectedColor =
                Cairo::SolidPattern::create_rgb( 0.5, 0.0, 0.0 );

        Glib::signal_timeout().connect(
                sigc::mem_fun(*this,&Main::queue_draw),
                500 );
    }


    void clear()
    {
        diagram.clear();
        //view.queue_draw();
    }

    bool queue_draw()
    {
        if(m_needsDraw)
        {
            m_needsDraw = false;
            view.queue_draw();
        }
        return true;
    }


    void onDraw( const Cairo::RefPtr<Cairo::Context>& ctx )
    {
        double s = std::max( adj_ws_w->get_value(), adj_ws_h->get_value() );
        double r = 0.002*s;   // circle radius for points
        double r2= 0.008;    // circle radius for horizon ridges
        double w = 0.001*s;   // line width
        double ns = 0.05;   // size of drawn normals
        double as = 0.01;   // length of arrow heads

        // draw origiin
        ctx->set_line_width(w);
        {
            double x,y,w,h;
            x = adj_ws_x->get_value();
            y = adj_ws_y->get_value();
            w = adj_ws_w->get_value();
            h = adj_ws_h->get_value();
            ctx->set_source_rgb(0,0,1);

            ctx->move_to( x, 0 );
            ctx->line_to( x+w, 0 );
            ctx->stroke();

            ctx->move_to( 0, y );
            ctx->line_to( 0, y+h );
            ctx->stroke();
        }


        Diagram_t::CellMap_t::const_iterator iCell;

        for(iCell = diagram.cells().begin();
                iCell != diagram.cells().end(); iCell++)
        {
            // draw the voronoi edges
            const Diagram_t::Cell_t& cell = iCell->second;
            ctx->set_source(edgeColor);
            ctx->set_line_width(w);
            drawCell( ctx, cell );

            // draw the site
            const Diagram_t::Point_t& p = *iCell->first;
            ctx->move_to( (double) p[0] + r,
                          (double) p[1]);
            ctx->arc( (double) p[0],
                      (double) p[1],
                      r,0,2*M_PI);
            ctx->set_source(siteColor);
            ctx->fill();
        }

        // fill the selected cells
        ctx->set_source(selectedColor);
        ctx->set_line_width(2*w);
        std::set< const Diagram_t::Cell_t* >::iterator iCell2;
        for( iCell2 = m_selectedCells.begin();
                iCell2 != m_selectedCells.end();
                ++iCell2 )
        {
            fillCell( ctx, **iCell2 );
        }

        // if there is an active cell then highlight it
        if( chk_draw_active->get_active() )
        {
            ctx->set_source(activeColor);
            ctx->set_line_width(2*w);
            if( m_activeCell )
                fillCell( ctx, *m_activeCell );
        }
    }

    void onSave()
    {
        Gtk::FileChooserDialog dialog("Please choose a file",
                Gtk::FILE_CHOOSER_ACTION_SAVE);
        dialog.set_transient_for(*win_main);

        //Add response buttons the the dialog:
        dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
        dialog.add_button(Gtk::Stock::OPEN, Gtk::RESPONSE_OK);

        //Add filters, so that only certain file types can be selected:

        Glib::RefPtr<Gtk::FileFilter> filter_text = Gtk::FileFilter::create();
        filter_text->set_name("Text files");
        filter_text->add_mime_type("text/plain");
        dialog.add_filter(filter_text);

        Glib::RefPtr<Gtk::FileFilter> filter_any = Gtk::FileFilter::create();
        filter_any->set_name("Any files");
        filter_any->add_pattern("*");
        dialog.add_filter(filter_any);

        //Show the dialog and wait for a user response:
        int result = dialog.run();

        //Handle the response:
        switch(result)
        {
            case(Gtk::RESPONSE_OK):
            {
                std::string filename = dialog.get_filename();
                std::ofstream ofile;
                ofile.open(filename.c_str());
                if(!ofile.is_open())
                {
                    std::cout << "Failed to open " << filename << " for write"
                              << std::endl;
                    break;
                }

                std::set< const Diagram_t::Cell_t* >::iterator iCell;
                for( iCell = m_selectedCells.begin();
                        iCell != m_selectedCells.end();
                        ++iCell )
                {
                    // put the vertices in an ordered set
                    typedef std::map< double, const Diagram_t::Point_t* > Ordered_t;

                    Ordered_t vertices;
                    Diagram_t::Cell_t::PointSet_t::const_iterator ipPoint;
                    for( ipPoint = (*iCell)->vertices.begin();
                            ipPoint != (*iCell)->vertices.end();
                            ++ipPoint )
                    {
                        Diagram_t::Vector_t v =  ((**ipPoint) - *((*iCell)->site) );
                        double a = std::atan2( (double) v[1], (double) v[0] );
                        vertices[a] = *ipPoint;
                    }

                    // now write out that set as parenthesis pairs
                    for( Ordered_t::iterator iVertex = vertices.begin();
                            iVertex != vertices.end();
                            ++iVertex )
                    {
                        const Diagram_t::Point_t& p= *(iVertex->second);
                        ofile << "("
                                   << p[0]
                                   << ","
                                   << p[1]
                                   << "); ";
                    }
                    ofile << "\n";
                }

                break;
            }
        }
    }

    void saveImage()
    {
        Gtk::FileChooserDialog dialog("Save Image",
                Gtk::FILE_CHOOSER_ACTION_SAVE);
        dialog.set_transient_for(*win_main);

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
                    int w = 400;
                    int h = 400;
                    double xoff     = adj_ws_w->get_value();
                    double yoff     = adj_ws_h->get_value();
                    double xscale   = w/xoff;
                    double yscale   = h/yoff;

                    Cairo::RefPtr<Cairo::ImageSurface> img =
                        Cairo::ImageSurface::create(Cairo::FORMAT_ARGB32,w,h);
                    Cairo::RefPtr<Cairo::Context> ctx =
                            Cairo::Context::create(img);

                    ctx->scale(xscale,-yscale);
                    ctx->translate(0,-yoff);
                    ctx->set_source_rgb(1,1,1);
                    ctx->paint();

                    onDraw(ctx);

                    img->write_to_png(filename);
                }
                else if( ext.compare(".pdf") == 0)
                {
                    int w = 100;
                    int h = 100;
                    double xoff     = adj_ws_w->get_value();
                    double yoff     = adj_ws_h->get_value();
                    double xscale   = w/xoff;
                    double yscale   = h/yoff;
                    Cairo::RefPtr<Cairo::PdfSurface> img =
                        Cairo::PdfSurface::create(filename,w,h);
                    Cairo::RefPtr<Cairo::Context> ctx =
                            Cairo::Context::create(img);

                    ctx->scale(xscale,-yscale);
                    ctx->translate(0,-yoff);
                    ctx->set_source_rgb(1,1,1);
                    ctx->paint();

                    onDraw(ctx);
                }
                else if( ext.compare(".svg") == 0)
                {
                    int w = 100;
                    int h = 100;
                    double xoff     = adj_ws_w->get_value();
                    double yoff     = adj_ws_h->get_value();
                    double xscale   = w/xoff;
                    double yscale   = h/yoff;
                    Cairo::RefPtr<Cairo::SvgSurface> img =
                        Cairo::SvgSurface::create(filename,w,h);
                    Cairo::RefPtr<Cairo::Context> ctx =
                            Cairo::Context::create(img);

                    ctx->scale(xscale,-yscale);
                    ctx->translate(0,-yoff);
                    ctx->set_source_rgb(1,1,1);
                    ctx->paint();

                    onDraw(ctx);
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


    Gtk::Main gtkmm;
    Main handler;

    try
    {
        Gtk::Main::run(*handler.win_main);
        return 0;
    }
    catch( std::exception& e )
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    catch( Glib::Exception& e )
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 1;
}






