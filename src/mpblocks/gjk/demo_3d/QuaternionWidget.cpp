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
 *  @file   /home/josh/Codes/cpp/mpblocks2/examples/manipulator/workspace_cert/src/QuaternionWidget.cpp
 *
 *  @date   Sep 11, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */



#include "QuaternionWidget.h"
#include <iostream>
#include <cmath>

namespace mpblocks {
namespace    gjk88 {
namespace  demo_3d {

QuaternionWidget::QuaternionWidget()
{
    add_events( Gdk::BUTTON_MOTION_MASK | Gdk::POINTER_MOTION_MASK
              | Gdk::BUTTON_PRESS_MASK | Gdk::BUTTON_RELEASE_MASK );
    signal_motion_notify_event().connect(
            sigc::mem_fun(*this,&QuaternionWidget::on_mouse_move) );
    signal_button_release_event().connect(
            sigc::mem_fun(*this,&QuaternionWidget::on_mouse_release) );

    m_quat = Eigen::Quaternionf::Identity();
}

bool QuaternionWidget::on_mouse_move( GdkEventMotion* evt )
{
    if( evt->state & GDK_BUTTON1_MASK )
    {
        int h = get_allocated_height();
        int w = get_allocated_width();

        double x = evt->x / w;
        double y = 1 - (evt->y/h);
        double angle = std::atan2(y-0.5,x-0.5);
        if(m_active)
        {
            if( evt->state & Gdk::SHIFT_MASK )
            {
                double da = angle - m_angle;
                if( da > M_PI )
                    da -= 2*M_PI;
                if( da < -M_PI )
                    da += 2*M_PI;

                using namespace Eigen;
                Quaternionf qz( AngleAxisf( da, Vector3f::UnitZ() ) );
                m_quat = qz*m_quat;
                queue_draw();
                sig_value_changed();
            }
            else
            {
                double dx = x - m_x;
                double dy = y - m_y;

                using namespace Eigen;
                Quaternionf qy( AngleAxisf( 0.5f*dx*M_PI, Vector3f::UnitY() ) );
                Quaternionf qx( AngleAxisf( -0.5f*dy*M_PI, Vector3f::UnitX() ) );
                m_quat = qy*qx*m_quat;
                queue_draw();
                sig_value_changed();
            }
        }
        else
            m_active = true;

        m_x = x;
        m_y = y;
        m_angle = angle;
    }

    return true;
}

bool QuaternionWidget::on_mouse_release( GdkEventButton* evt )
{
    m_active = false;
    return true;
}

bool QuaternionWidget::on_draw( const Cairo::RefPtr<Cairo::Context>& ctx )
{
    // normalize the view
    int h = get_allocated_height();
    int w = get_allocated_width();
    ctx->scale(w,-h);
    ctx->translate(0,-1);
    ctx->rectangle(0,0,1,1);
    ctx->set_source_rgb(1,1,1);
    ctx->fill();

    // generate the four vectors we need
    using namespace Eigen;

    // initialize points of interested
    Vector4f X(1,0,0,1);
    Vector4f Y(0,1,0,1);
    Vector4f Z(0,0,1,1);

    // compute view matrix
    Matrix4f V = Matrix4f::Identity();
    V.block(0,0,3,3) = m_quat.toRotationMatrix();
    V.block(0,3,3,1) = Vector3f(0,0,-3);

    // compute projection matrix from quaternion
    float aspectRatio = float(w)/float(h);
    float yScale = std::tan(M_PI/2 - 45*M_PI/4);    //cotan(45deg/2)
    float xScale = yScale/aspectRatio;
    float zNear  = 0.03f;
    float zFar   = 100.0f;

    Matrix4f P;
    P << xScale, 0, 0, 0,
        0, yScale, 0, 0,
        0, 0, -(zFar+zNear)/(zFar-zNear), -1,
        0, 0, -2*zNear*zFar/(zFar-zNear), 0;

    // transform vertices
    Vector4f Xt = P.transpose()*V*X;
    Vector4f Yt = P.transpose()*V*Y;
    Vector4f Zt = P.transpose()*V*Z;

    // now draw the lines
    ctx->set_source_rgb(1,0,0);
    ctx->move_to(0.5,0.5);
    ctx->rel_line_to(Xt[0]/Xt[3],Xt[1]/Xt[3]);
    ctx->set_line_width(0.01);
    ctx->stroke();

    ctx->set_source_rgb(0,1,0);
    ctx->move_to(0.5,0.5);
    ctx->rel_line_to(Yt[0]/Yt[3],Yt[1]/Yt[3]);
    ctx->set_line_width(0.01);
    ctx->stroke();

    ctx->set_source_rgb(0,0,1);
    ctx->move_to(0.5,0.5);
    ctx->rel_line_to(Zt[0]/Zt[3],Zt[1]/Zt[3]);
    ctx->set_line_width(0.01);
    ctx->stroke();


    return true;
}

sigc::signal<void> QuaternionWidget::signal_value_changed()
{
    return sig_value_changed;
}

Eigen::Quaternionf QuaternionWidget::get_value()
{
    return m_quat;
}

void QuaternionWidget::set_value( const Eigen::Quaternionf& R, bool supress )
{
    m_quat = R.normalized();
    queue_draw();
    if( !supress )
        sig_value_changed();
}



} //< namespace mwc
} //< namespace examples
} //< namespace mpblocks


