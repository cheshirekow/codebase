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

        int         m_whichPoly;

        Eigen::Vector2d m_storedPos;

        Eigen::Vector2d m_pos[2];
        double          m_rot[2];

        std::vector<Eigen::Vector2d> m_poly[2];
        std::vector<Eigen::Vector2d> m_norm[2];
        double m_scale[2];

        gjk88::Result   m_gjkResult;

        typedef gjk88::PairVec<Eigen::Vector2d> Simplex;
        std::vector<Simplex> m_sHistory;
        std::vector<Eigen::Vector2d> m_vHistory;
        Eigen::Vector2d m_gjk[3];

    public:
        Main();
        virtual ~Main();

        void generatePoly( int which, int count );
        void configureGui();
        virtual void run();

        void whichPolyChanged();
        void polySizeChanged();
        void collisionTest();
        bool mouse_motion( GdkEventMotion* evt );
        bool mouse_press( GdkEventButton* evt );
        void draw( const Cairo::RefPtr<Cairo::Context>& ctx );
        void saveImage();
};


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

struct PointOps
{
    typedef Eigen::Vector2d Point;

    Point origin()
    {
        return Point(0,0);
    }

    // for 2-d embedding this is easy
    Eigen::Vector3d baryCentric(
            const Point& a, const Point& b, const Point& c )
    {
        Eigen::Matrix2d A;
        A.col(0) = b-a;
        A.col(1) = c-a;
        Eigen::Vector2d lambda = A.inverse()*(-a);
        return Eigen::Vector3d( 1-lambda[0]-lambda[1], lambda[0], lambda[1] );
    }

    bool containsOriginOld(
            const Point& a, const Point& b, const Point& c )
    {
        auto  lambda = baryCentric(a,b,c);
        return ( lambda[0] > 0 && lambda[1] > 0 && lambda[2] > 0 );
    }

    // for 2-d embedding this is easy
    bool containsOrigin(
            const Point& a, const Point& b, const Point& c )
    {
        Point ab = b-a;
        Point ac = c-a;
        Point ao = -a;
        Eigen::Vector3d b3(ab[0],ab[1],0);
        Eigen::Vector3d c3(ac[0],ac[1],0);
        Eigen::Vector3d o3(ao[0],ao[1],0);

        int bo = sgn( b3.cross(o3)[2] );
        int oc = sgn( o3.cross(c3)[2] );
        int bc = sgn( b3.cross(c3)[2] );

        return( bo == oc && oc == bc );
    }

    Point normalize( const Point& a )
    {
        return a.normalized();
    }

    double squaredNorm( const Point& a )
    {
        return a.squaredNorm();
    }

    double dot( const Point& a, const Point& b )
    {
        return a.dot(b);
    }

    bool threshold( const Point& a, const Point& b )
    {
        return false;
    }

    bool threshold( const Point& a, const Point& b, const Point& c )
    {
        return true;
    }
};

struct SupportFn
{
    typedef Eigen::Vector2d Point;

    const std::vector<Eigen::Vector2d>& m_A;
    const std::vector<Eigen::Vector2d>& m_B;
    Eigen::Vector2d m_xA;
    Eigen::Vector2d m_xB;
    double m_rA;
    double m_rB;

    SupportFn( const std::vector<Eigen::Vector2d>& A,
            const std::vector<Eigen::Vector2d>& B,
            Eigen::Vector2d xA,
            Eigen::Vector2d xB,
            double rA,
            double rB):
        m_A(A),
        m_B(B),
        m_xA(xA),
        m_xB(xB),
        m_rA(rA),
        m_rB(rB)
    {}


    void operator()( const Point& dir, Point& a, Point& b )
    {
        Eigen::Vector2d d( dir[0], dir[1] );
        double max = -1e6;
        for( const Eigen::Vector2d& pt : m_A )
        {
            Eigen::Vector2d x = m_xA + Eigen::Rotation2Dd(m_rA)*pt;
            if( d.dot(x) > max )
            {
                a << x[0], x[1];
                max = d.dot(x);
            }
        }

        max = -1e6;
        for( const Eigen::Vector2d& pt : m_B )
        {
            Eigen::Vector2d x = m_xB + Eigen::Rotation2Dd(m_rB)*pt;
            if( d.dot(-x) > max )
            {
                b << x[0], x[1];
                max = d.dot(-x);
            }
        }
    }

};


} //< namespace demo
} //< namespace gjk88
} //< namespace mpblocks




















#endif // MAIN_H_
