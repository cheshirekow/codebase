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
 *  \file   ViewArea.cpp
 *
 *  \date   Aug 15, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include "ViewArea.h"


namespace                   mpblocks {
namespace            voronoi_diagram {
namespace two_d_obstacle_set_creator {



ViewArea::ViewArea()
{
    m_sideLength = 1.0;
    m_x = m_y = 0;
    m_w = m_h = 1.0;
    this->add_events( Gdk::POINTER_MOTION_MASK
                        |  Gdk::BUTTON_PRESS_MASK
                        |  Gdk::BUTTON_RELEASE_MASK
                        |  Gdk::BUTTON1_MOTION_MASK);
}


void ViewArea::setArea( double x, double y, double w, double h )
{
    m_x = x;
    m_y = y;
    m_w = w;
    m_h = h;
    queue_draw();
}

bool ViewArea::on_draw(const Cairo::RefPtr<Cairo::Context>& ctx)
{
    int width       = this->get_allocated_width();
    int height      = this->get_allocated_height();
    double xscale   = width/m_w;
    double yscale   = height/m_h;

    // first draw an outline
    ctx->rectangle(0,0,width,height);
    ctx->set_source_rgb(1,1,1);
    ctx->fill_preserve();
    ctx->set_line_width(2);
    ctx->set_source_rgb(0,0,0);
    ctx->stroke();

    // save state
    ctx->save();

    // now flip coordinates and scale so that the bottom left is 0,0
    // and the top right is (s,s)
    ctx->scale(xscale,-yscale);
    ctx->translate(0,-m_h);
    ctx->translate(-m_x,-m_y);

    // then let everything draw itself
    sig_draw.emit(ctx);

    ctx->restore();

    return true;
}





} // namespace two_d_obstacle_set_creator
} // namespace voronoi_diagram
} // namespace mpblocks 
