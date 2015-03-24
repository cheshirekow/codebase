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
 *  \file   ViewArea.h
 *
 *  \date   Aug 15, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef MPBLOCKS_VORONOI_DIAGRAM_2D_OBSTACLE_SET_GENERATOR_VIEWAREA_H_
#define MPBLOCKS_VORONOI_DIAGRAM_2D_OBSTACLE_SET_GENERATOR_VIEWAREA_H_

#include <gtkmm.h>
#include <iostream>
#include <string>



namespace                   mpblocks {
namespace            voronoi_diagram {
namespace two_d_obstacle_set_creator {





class ViewArea :
    public Gtk::DrawingArea
{
    private:
        double m_sideLength;
        double m_x;
        double m_y;
        double m_w;
        double m_h;

    public:
        ViewArea();
        void setArea( double x, double y, double w, double h );
        sigc::signal<void,const Cairo::RefPtr<Cairo::Context>&> sig_draw;

    virtual bool on_draw(const Cairo::RefPtr<Cairo::Context>&);
};





} // namespace two_d_obstacle_set_creator
} // namespace voronoi_diagram
} // namespace mpblocks 

#endif // VIEWAREA_H_
