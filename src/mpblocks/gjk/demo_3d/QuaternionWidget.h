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
 *  @file   /home/josh/Codes/cpp/mpblocks2/examples/manipulator/workspace_cert/src/QuaternionWidget.h
 *
 *  @date   Sep 11, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_QUATERNIONWIDGET_H_
#define MPBLOCKS_QUATERNIONWIDGET_H_

#include <gtkmm.h>
#include <Eigen/Dense>

namespace mpblocks {
namespace    gjk88 {
namespace  demo_3d {


class QuaternionWidget:
    public Gtk::DrawingArea
{
    private:
        Eigen::Quaternionf m_quat;
        sigc::signal<void> sig_value_changed;

        bool   m_active;
        double m_x;
        double m_y;
        double m_angle;

    public:
        QuaternionWidget();
        sigc::signal<void> signal_value_changed();
        Eigen::Quaternionf get_value();
        void set_value( const Eigen::Quaternionf& R, bool supress=false );

    private:
        bool on_mouse_move( GdkEventMotion* evt );
        bool on_mouse_release( GdkEventButton* evt );
        virtual bool on_draw( const Cairo::RefPtr<Cairo::Context>& ctx );
};




} //< namespace mwc
} //< namespace examples
} //< namespace mpblocks






#endif // QUATERNIONWIDGET_H_
