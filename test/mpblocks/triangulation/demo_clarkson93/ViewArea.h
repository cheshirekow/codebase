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

#ifndef MPBLOCKS_CONVEX_HULL_DEMO_VIEWAREA_H_
#define MPBLOCKS_CONVEX_HULL_DEMO_VIEWAREA_H_

#include <gtkmm.h>
#include <iostream>
#include <string>



namespace    mpblocks {
namespace convex_hull {
namespace        demo {





class ViewArea :
    public Gtk::DrawingArea
{
    public:
        ViewArea();
        sigc::signal<void,const Cairo::RefPtr<Cairo::Context>&> sig_draw;

    virtual bool on_draw(const Cairo::RefPtr<Cairo::Context>&);
};





} // namespace demo 
} // namespace convex_hull 
} // namespace mpblocks 

#endif // VIEWAREA_H_
