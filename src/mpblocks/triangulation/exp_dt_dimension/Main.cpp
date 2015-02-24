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
 *  @file   /home/josh/Codes/cpp/mpblocks2/triangulation/src/exp_dt_dimension/Main.cpp
 *
 *  @date   Aug 8, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */


#include "Main.h"


namespace         mpblocks {
namespace exp_dt_dimension {


Main::Main()
{
    m_layoutFile = std::string(g_srcDir) + "/layout.glade";
    m_stateFile  = std::string(g_binDir) + "/state.yaml";

    m_layout.loadLayout(m_layoutFile);
    for(unsigned int i=0; i < 4*NEXP; i++)
    {
        std::stringstream str;
        str << "chk_" << i;
        m_enabled[i].set_name(str.str());
        m_layout.set<Gtk::CheckButton>(str.str(),m_enabled+i);
    }

    m_layout.widget<Gtk::AspectFrame>("viewFrame")->add(m_view);
    m_view.show();

    const char* keys[] = {
            "chk_sites",     "chk_delaunay", "chk_balls",
            "chk_vpoints",   "chk_edges",    "chk_weights" };
    for (int i = 0; i < 5; i++)
    {
        m_layout.get<Gtk::CheckButton>(keys[i])->signal_clicked().connect(
                sigc::mem_fun(m_view, &gtk::SimpleView::queue_draw));
    }

    m_layout.get<Gtk::Button>("saveImage")->signal_clicked().connect(
            sigc::mem_fun(this, &Main::saveImage));

    m_layout.get<Gtk::Adjustment>("edgeWidth")
            ->signal_value_changed().connect(
                    sigc::mem_fun(m_view,&gtk::SimpleView::queue_draw) );
    m_layout.get<Gtk::Adjustment>("pointRadius")
            ->signal_value_changed().connect(
                    sigc::mem_fun(m_view,&gtk::SimpleView::queue_draw) );

    m_layout.get<Gtk::ToggleButton>("run")->signal_clicked().
            connect( sigc::mem_fun(*this,&Main::runToggled));
    m_layout.get<Gtk::Button>("reset")->signal_clicked().
            connect( sigc::mem_fun(*this,&Main::resetPressed));

    m_layout.get<Gtk::Button>("selectOutDir")->signal_clicked()
            .connect( std::bind(&Main::setOutDir,this) );
    m_layout.get<Gtk::Button>("saveData")->signal_clicked()
            .connect( std::bind(&Main::saveData,this) );


    m_layout.get<Gtk::Adjustment>("nSamples")->signal_value_changed()
            .connect( sigc::bind(
                    sigc::mem_fun(*this,&Main::paramChanged),
                    Runner::KEY_STEPS, "nSamples" ) );

    m_layout.get<Gtk::Adjustment>("nSimplices")->signal_value_changed()
            .connect( sigc::bind(
                    sigc::mem_fun(*this,&Main::paramChanged),
                    Runner::KEY_SIMPLICES, "nSimplices" ) );

    m_layout.get<Gtk::Adjustment>("nBatch")->signal_value_changed()
            .connect( sigc::bind(
                    sigc::mem_fun(*this,&Main::paramChanged),
                    Runner::KEY_BATCH, "nBatch" ) );

    m_layout.get<Gtk::Adjustment>("nTrials")->signal_value_changed()
            .connect( sigc::bind(
                    sigc::mem_fun(*this,&Main::paramChanged),
                    Runner::KEY_TRIALS, "nTrials" ) );

    Gtk::Grid* grid = m_layout.get<Gtk::Grid>("expGrid");

    /// add labels for dimension and experiment type
    for(unsigned int i=0; i < 4; i++)
    {
        std::stringstream txt[NEXP];
        for(unsigned int j=0; j < NEXP; j++)
            txt[j] << (2+j);

        int top  = 4;
        int left = 2;
        for(unsigned int j=0; j < NEXP; j++)
        {
            grid->attach(m_dimLabels[4*i+j],left,top+5*i+j,1,1);
            m_dimLabels[4*i+j].set_text(txt[j].str());
        }

        left = 3;
        for(unsigned int j=0; j < NEXP; j++)
            grid->attach(m_enabled[4*i+j],left,top+5*i+j,1,1);
        left = 4;
        for(unsigned int j=0; j < NEXP; j++)
            grid->attach(m_trialProgress[4*i+j],left,top+5*i+j,1,1);
        left = 5;
        for(unsigned int j=0; j < NEXP; j++)
            grid->attach(m_totalProgress[4*i+j],left,top+5*i+j,1,1);
    }
    grid->show_all_children(true);

    // monospace font
//            Pango::FontDescription fontdesc("monospace");
//            m_layout.widget<Gtk::TextView>("depthView")
//                    ->override_font(fontdesc);
    constructExperiments();

    for(unsigned int i=0; i < 4*NEXP; i++)
        m_run[i].setExperiment(m_exp[i]);
}

Main::~Main()
{
    for(unsigned int i=0; i < 4*NEXP; i++)
        delete m_exp[i];
}

bool Main::getProgress()
{
    for(unsigned int i=0; i < 4*NEXP; i++)
    {
        m_trialProgress[i].set_fraction( m_run[i].trialProgress() );
        m_totalProgress[i].set_fraction( m_run[i].totalProgress() );
    }

    return true;
}

void Main::run()
{
    Glib::signal_timeout().connect(
            sigc::mem_fun(*this,&Main::getProgress), 100 );
    m_layout.loadValues(m_stateFile);
    m_outDir = m_layout.get<Gtk::EntryBuffer>("outDir")->get_text();
    m_gtkmm.run(*m_layout.widget<Gtk::Window>("main"));
    m_layout.saveValues(m_stateFile);
}

void Main::killWorkers()
{
    for(unsigned int i=0; i < 4*NEXP; i++)
        m_run[i].quit();

    for(unsigned int i=0; i < 4*NEXP; i++)
        m_run[i].join();
}

void Main::runToggled()
{
    killWorkers();

    if( m_layout.get<Gtk::ToggleButton>("run")->get_active() )
    {
        for(unsigned int i=0; i < 4*NEXP; i++)
        {
            if( m_enabled[i].get_active() )
                m_run[i].launch();
        }

        m_layout.get<Gtk::ToggleButton>("run")->
            set_label("running... click to pause");
    }
    else
        m_layout.get<Gtk::ToggleButton>("run")->set_label("continue");

    std::cout << "exiting from runToggled" << std::endl;
}

void Main::paramChanged( Runner::Key key, const std::string& adj )
{
    int val = m_layout.get<Gtk::Adjustment>(adj)->get_value();

    for(unsigned int i=0; i < 4*NEXP; i++)
         m_run[i].set(key,val);
}

void Main::resetPressed()
{
    killWorkers();
    for(unsigned int i=0; i < 4*NEXP; i++)
        m_run[i].reset();

    m_layout.get<Gtk::ToggleButton>("run")->set_active(false);
    m_layout.get<Gtk::ToggleButton>("run")->set_label("start");
}


void Main::draw(const Cairo::RefPtr<Cairo::Context>& ctx)
{
//            double eWidth = m_layout.get<Gtk::Adjustment>("edgeWidth")->get_value()/20000.0;
//            double rPoint = m_layout.get<Gtk::Adjustment>("pointRadius")->get_value()/10000.0;
//
//
//            gtk::EigenCairo ectx(ctx);
//            ctx->set_line_width(eWidth);
//
//            if( m_layout.get<Gtk::CheckButton>("chk_sites")->get_active() )
//            {
//                Gdk::RGBA color =
//                    m_layout.get<Gtk::ColorButton>("color_sites")->get_rgba();
//                ectx.set_source_rgba(
//                        color.get_red(),
//                        color.get_green(),
//                        color.get_blue(),
//                        color.get_alpha() );
//                for( const Point& p : m_ptStore )
//                {
//                    Point x0 = m_inv(p);
//                    Eigen::Matrix<double,2,1> x1(x0[0],x0[1]);
//                    ectx.circle(x1,rPoint);
//                    ectx.fill();
//                }
//            }
//
//            if( m_ptStore.size() < 4)
//                return;
//
//            bool drawDelaunay =
//                    m_layout.get<Gtk::CheckButton>("chk_delaunay")->get_active();
//            bool drawWeights =
//                    m_layout.get<Gtk::CheckButton>("chk_weights")->get_active();
//            bool drawCircumCenter =
//                    m_layout.get<Gtk::CheckButton>("chk_vpoints")->get_active();
//            bool drawCircumSphere =
//                    m_layout.get<Gtk::CheckButton>("chk_balls")->get_active();
//            bool drawVoronoi =
//                    m_layout.get<Gtk::CheckButton>("chk_edges")->get_active();
//
//            m_hull_Q.clear();
//            m_hull_E.clear();
//
//            // start the stack at some hull simplex
//            m_hull_Q.push(m_T.m_hullSimplex);
//
//            // pointer to the point buffer
//            Point* pts = m_ptStore.data();
//
//            PointDeref deref(&m_ptStore);
//            while (m_hull_Q.size() > 0)
//            {
//                // pop a simplex off the stack
//                Simplex* S = m_hull_Q.pop();
//
//                // it is a nearest site if the inversion center is not
//                // visible, so draw it
//                if (!S->isVisible(m_inv.center()))
//                {
//                    if( drawDelaunay || drawWeights )
//                    {
//                        Point x0 = m_inv(pts[S->V[3]]);
//                        ectx.move_to((double) x0[0], (double) x0[1]);
//                        for (int i = 1; i < 4; i++)
//                        {
//                            Point x1 = m_inv(pts[S->V[i]]);
//                            ectx.line_to((double) x1[0], (double) x1[1]);
//                        }
//
//                        if( drawWeights )
//                        {
//                            Gdk::RGBA color =
//                                m_layout.get<Gtk::ColorButton>("color_weights")->get_rgba();
//                            ectx.set_source_rgba(
//                                    color.get_red(),
//                                    color.get_green(),
//                                    color.get_blue(),
//                                    S->weight / m_btps.sum() );
//                            ectx.fill_preserve();
//                        }
//
//                        if( drawDelaunay )
//                        {
//                            Gdk::RGBA color =
//                                m_layout.get<Gtk::ColorButton>("color_delaunay")->get_rgba();
//                            ectx.set_source_rgba(
//                                    color.get_red(),
//                                    color.get_green(),
//                                    color.get_blue(),
//                                    color.get_alpha() );
//                            ectx.stroke();
//                        }
//                    }
//
//                    computeCenter(m_inv,deref,S);
//
//                    if( drawCircumCenter || drawCircumSphere )
//                    {
//                        if( drawCircumSphere )
//                        {
//                            Gdk::RGBA color =
//                                m_layout.get<Gtk::ColorButton>("color_balls")->get_rgba();
//                            ectx.set_source_rgba(
//                                    color.get_red(),
//                                    color.get_green(),
//                                    color.get_blue(),
//                                    color.get_alpha() );
//                            ectx.circle(S->c_c,S->c_r);
//                            ectx.fill();
//                        }
//
//                        if( drawCircumCenter )
//                        {
//                            Gdk::RGBA color =
//                                m_layout.get<Gtk::ColorButton>("color_vpoints")->get_rgba();
//                            ectx.set_source_rgba(
//                                    color.get_red(),
//                                    color.get_green(),
//                                    color.get_blue(),
//                                    color.get_alpha() );
//                            ectx.circle(S->c_c,rPoint);
//                            ectx.fill();
//                        }
//                    }
//
//                    if( drawVoronoi )
//                    {
//                        for(int i=1; i < 4; i++)
//                        {
//                            Simplex* N = S->N[i];
//                            if( !N->isVisible( m_inv.center() ) )
//                            {
//                                computeCenter(m_inv,deref,N);
//                                Gdk::RGBA color =
//                                m_layout.get<Gtk::ColorButton>("color_edges")->get_rgba();
//                                ectx.set_source_rgba(
//                                        color.get_red(),
//                                        color.get_green(),
//                                        color.get_blue(),
//                                        color.get_alpha() );
//
//                                ectx.move_to(S->c_c);
//                                ectx.line_to(N->c_c);
//                                ectx.stroke();
//                            }
//                        }
//                    }
//                }
//
//                // in any case, expand it
//                m_hull_E.push(S);
//
//                // get all it's infinite neighbors
//                for (int i = 1; i < 4; i++)
//                {
//                    Simplex* N = S->N[i];
//                    // if we haven't already queued or expanded it then queue
//                    // it
//                    if (!m_hull_Q.isMember(N) && !m_hull_E.isMember(N))
//                        m_hull_Q.push(N);
//                }
//            }
}


void Main::saveImage()
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


void Main::setOutDir()
{
    Gtk::FileChooserDialog dialog("Select output directory",
            Gtk::FILE_CHOOSER_ACTION_SELECT_FOLDER);
    dialog.set_transient_for(*m_layout.get<Gtk::Window>("main"));

    //Add response buttons the the dialog:
    dialog.add_button(Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL);
    dialog.add_button(Gtk::Stock::OPEN,   Gtk::RESPONSE_OK);
    dialog.set_filename(m_outDir);

    //Show the dialog and wait for a user response:
    int result = dialog.run();

    //Handle the response:
    if( result == Gtk::RESPONSE_OK )
    {
        m_outDir = dialog.get_filename();
        m_layout.get<Gtk::EntryBuffer>("outDir")->set_text(m_outDir);
    }
}

void Main::saveData()
{
    m_layout.saveValues( m_outDir + "/experiment.yaml" );
    for(unsigned int i=0; i < 4*NEXP; i++)
        if(m_enabled[i].get_active())
            m_exp[i]->save(m_outDir);
}


} //< namespace exp_dt_dimension
} //< namespace mpblocks

