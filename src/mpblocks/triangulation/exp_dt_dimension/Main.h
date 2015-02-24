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
 *  @file   /home/josh/Codes/cpp/mpblocks2/triangulation/src/exp_dt_dimension/Main.h
 *
 *  @date   Aug 8, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_EXP_DT_DIMENSION_MAIN_H_
#define MPBLOCKS_EXP_DT_DIMENSION_MAIN_H_


#include "config.h"
#include "ExperimentBase.h"
#include "Runner.h"

#include <mpblocks/gtk.hpp>
#include <cpp_pthreads.h>

#include <gtkmm.h>
#include <boost/format.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <functional>




namespace         mpblocks {
namespace exp_dt_dimension {

static const unsigned int NEXP = 4;





struct Main
{
    private:
        std::string m_layoutFile;
        std::string m_stateFile;

        Gtk::Main       m_gtkmm;
        gtk::LayoutMap  m_layout;
        gtk::SimpleView m_view;

        Gtk::Label          m_dimLabels[4*NEXP];
        Gtk::CheckButton    m_enabled[4*NEXP];
        Gtk::ProgressBar    m_trialProgress[4*NEXP];
        Gtk::ProgressBar    m_totalProgress[4*NEXP];

        ExperimentBase*     m_exp[4*NEXP];
        Runner              m_run[4*NEXP];

        pthreads::Thread    m_serialThread;
        std::string         m_outDir;

    public:
        void constructExperiments();
        Main();
        ~Main();

        bool getProgress();
        void run();
        void killWorkers();
        void runToggled();
        void resetPressed();
        void paramChanged( Runner::Key key, const std::string& adj );
        void draw(const Cairo::RefPtr<Cairo::Context>& ctx);
        void saveImage();
        void setOutDir();
        void saveData();
};


} //< namespace exp_dt_dimension
} //< namespace mpblocks












#endif // MAIN_H_
