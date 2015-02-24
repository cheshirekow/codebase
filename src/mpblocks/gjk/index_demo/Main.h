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
 *  @file   /home/josh/Codes/cpp/mpblocks2/gjk/test/demo/Main.h
 *
 *  @date   Sep 15, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_MAIN_H_
#define MPBLOCKS_MAIN_H_

#include "MainIface.h"

#include <gtkmm.h>
#include <Eigen/Dense>
#include <mpblocks/gtk.h>
#include <mpblocks/gjk88.h>

namespace mpblocks {
namespace    gjk88 {
namespace     demo {

class Main:
    public MainIface
{
    private:
        Gtk::Main m_gtkmm;

        std::string m_gladeFile;
        std::string m_yamlFile;

        gtk::LayoutMap  m_layout;
        gtk::SimpleView m_view;
        gtk::SimpleView m_view2;

        Eigen::Vector2d m_storedPos;

        std::vector<Eigen::Vector4d> m_vertices;
        std::vector<Eigen::Vector3i> m_faces;

        // maps pairs of vertices to pairs of faces
        std::map< std::pair<int,int>, std::vector<int> >    m_edges;

        double m_dist;
        double m_vAngle;
        double m_hAngle;

        Eigen::Matrix4d m_PV;

    public:
        Main();
        virtual ~Main();

        void configureGui();
        virtual void run();

        void queueDraw();
        void cameraChanged();
        bool mouse_motion( GdkEventMotion* evt );
        bool mouse_press( GdkEventButton* evt );
        void draw( const Cairo::RefPtr<Cairo::Context>& ctx );
        void draw2( const Cairo::RefPtr<Cairo::Context>& ctx );

        void loadObject();
        void objectFileChanged();
        void saveImage();
};



} //< namespace demo
} //< namespace gjk88
} //< namespace mpblocks




















#endif // MAIN_H_
