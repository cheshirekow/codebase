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


#include "Experiment.h"
#include "Main.h"

namespace         mpblocks {
namespace exp_dt_dimension {

void Main::constructExperiments()
{
    m_exp[0] = new Experiment<float,2,false>();
    m_exp[1] = new Experiment<float,3,false>();
    m_exp[2] = new Experiment<float,4,false>();
    m_exp[3] = new Experiment<float,5,false>();

    m_exp[4] = new Experiment<float,2,true >();
    m_exp[5] = new Experiment<float,3,true >();
    m_exp[6] = new Experiment<float,4,true >();
    m_exp[7] = new Experiment<float,5,true >();

    m_exp[ 8] = new Experiment<double,2,false>();
    m_exp[ 9] = new Experiment<double,3,false>();
    m_exp[10] = new Experiment<double,4,false>();
    m_exp[11] = new Experiment<double,5,false>();

    m_exp[12] = new Experiment<double,2,true >();
    m_exp[13] = new Experiment<double,3,true >();
    m_exp[14] = new Experiment<double,4,true >();
    m_exp[15] = new Experiment<double,5,true >();
}

} //< namespace exp_dt_dimension
} //< namespace mpblocks



using namespace mpblocks::exp_dt_dimension;

int main(int argc, char** argv)
{
    Main m_app;
    m_app.run();
    return 0;
}

