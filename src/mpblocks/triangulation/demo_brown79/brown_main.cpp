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
 *  \file   test/demo/main.cpp
 *
 *  \date   Oct 25, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include "config.h"

#include <gtkmm.h>
#include <mpblocks/brown79.hpp>
#include <mpblocks/gtk.hpp>


namespace mpblocks {
namespace delaunay {
namespace     test {






/// documents the interface for Traits : encapsulates various policies for the
/// Triangulation
/**
 *  @todo   Should these traits be split up into multiple policy classes?
 */
struct ClarksonTraits
{
    typedef unsigned int uint;

    /// dimension of the embeded space, use clarkson93::Dynamic for a
    /// datastructure whose dimension is set at runtime
    static const unsigned int NDim          = 3;

    /// number format for storing indices
    typedef unsigned int idx_t;

    /// number format for scalar numbers
    typedef double Scalar;

    /// Data structure for representing points.
    /**
     *  Currently only
     *  Eigen::Matrix<Scalar,NDim,1> is supported b/c we utilize this structure
     *  in solving a linear system. Technically anything that derives from
     *  Eigen::MatrixBase will work because it just needs to be assignable to
     *  an Eigen::Matrix::row() and from an Eigen::Vector when we solve for
     *  the normal and offset of the base facet. In the future I may generalize
     *  the solving of the linear system in which case that functionality will
     *  be a requirement of the traits class
     */
    typedef Eigen::Matrix<Scalar,NDim,1>   Point;

    /// a reference to a point
    typedef uint PointRef;

    /// proves a means of turning a PointRef into a Point&
    /**
     *  In the example traits PointRef is a Point* so we simply dereference this
     *  pointer. If PointRef were an index into an array of Point structures,
     *  then PointDeref should store a pointer to the beginning of that array.
     *  For example
     *
     *  @code
    typedef unsigned int PointRef;
    struct PointDeref
    {
        Point* m_buf;
        Point& operator()( PointRef i ){ return m_buf[i]; }
    };
@endcode
     */
    class PointDeref
    {
        private:
            std::vector<Point>* m_buf;

        public:
            PointDeref( std::vector<Point>* buf=0 ){ setBuf(buf); }
            void setBuf( std::vector<Point>* buf ) { m_buf = buf; }
            const Point& operator()(  PointRef idx ){ return (*m_buf)[idx]; }
    };

    /// The derived type to use for simplices.
    /**
     *  The only requirement is that it
     *  must derive from clarkson93::SimplexBase instantiated with the Traits
     *  class. Otherwise, it may contain as many extra data members as you want.
     *  You may hook into the triangulation control flow by overriding
     *  member functions of SimplexBase. If you do so your implementation will
     *  be called, instead of SimplexBase's (even though your implementation is
     *  not virtual). This is because the triangulation code maintains only
     *  Simplex* pointers (i.e. pointers to this type) not pointers to the
     *  base type. If you choose to hook into the program flow this way then
     *  you must be sure to call the SimplexBase member functions from your
     *  override.
     */
    struct Simplex:
        clarkson93::SimplexBase<ClarksonTraits>
    {
        // extra data members and member functions here
    };

    /// template for allocators & memory managers
    struct SimplexFactory:
        public std::vector<Simplex>
    {
        typedef std::vector<Simplex> Base;

        /// construct an object of type T using the next available memory
        // slot and return it's pointer
        Simplex* create()
        {
            assert( Base::size() < Base::capacity() );
            Base::emplace_back();
            return &Base::back();
        }
    };

    /// the triangulation provides some static callbacks for when hull faces
    /// are added or removed. If we wish to do anything special this is where
    /// we can hook into them. If you do not wish to hook into the callbacks
    /// then simply create an empty structure which has empty implementation
    /// for these
    struct Callback
    {
        void hullFaceAdded( Simplex* S ){}
        void hullFaceRemoved( Simplex* S ){}
    };

};



class View:
    public Gtk::DrawingArea
{
    public:
        sigc::signal<void,const Cairo::RefPtr<Cairo::Context>&> sig_draw;
        sigc::signal<void,double,double>                        sig_mouse;

        View()
        {
            this->add_events( Gdk::POINTER_MOTION_MASK |
                               Gdk::BUTTON_PRESS_MASK );

            this->signal_button_press_event().connect(
                    sigc::mem_fun(*this,&View::on_click) );
        }


        bool on_click( GdkEventButton* evt )
        {
            if( evt->button == 1 )
            {
                double x =       evt->x / this->get_allocated_width();
                double y = 1.0 - evt->y / this->get_allocated_height();
                sig_mouse.emit(x,y);
            }
            return true;
        }

        virtual bool on_draw( const Cairo::RefPtr<Cairo::Context>& ctx )
        {
            int width  = this->get_allocated_width();
            int height = this->get_allocated_height();

            // draw white background
            ctx->set_source_rgb(1,1,1);
            ctx->paint();

            // scale the context so (0,0) is bottom left and (1,1) is top
            // right
            ctx->scale(width,-height);
            ctx->translate(0,-1);

            sig_draw.emit(ctx);

            return true;
        }
};


typedef ClarksonTraits::Simplex                    Simplex;
typedef ClarksonTraits::Point                      Point;
typedef ClarksonTraits::PointRef                   PointRef;
typedef clarkson93::Triangulation<ClarksonTraits>  Triangulation;
typedef clarkson93::Stack<Simplex*,
                        clarkson93::SimplexBits>   SimplexStack;
typedef brown79::Inversion<ClarksonTraits>         Inversion;



class Main
{
    private:
        std::string m_layoutFile;
        std::string m_stateFile;

        gtk::LayoutMap   m_layout;
        Gtk::Main   m_gtkmm;
        View        m_view;

        Triangulation       m_T;
        Inversion           m_inv;
        SimplexStack        m_hull_Q;
        SimplexStack        m_hull_E;
        std::vector<Point>  m_ptStore;

    public:
        Main()
        {
            m_layoutFile = std::string(g_srcDir) +  "/layout.glade";
            m_stateFile  = std::string(g_binDir) +  "/state.yaml";

            m_layout.loadLayout( m_layoutFile );
            m_layout.loadValues( m_stateFile );

            m_layout.widget<Gtk::AspectFrame>("viewFrame")->add(m_view);
            m_view.show();


            sigc::slot<void,const Cairo::RefPtr<Cairo::Context>&>
                slot_draw = sigc::mem_fun(*this,&Main::draw);
            sigc::slot<void,double,double>
                slot_addPoint = sigc::mem_fun(*this,&Main::addPoint);

            m_view.sig_draw.connect( slot_draw );
            m_view.sig_mouse.connect( slot_addPoint );

            m_inv.init( Point(0.5,0.5,1), 1 );
            m_ptStore.reserve(1000);
            m_T.m_antiOrigin = -1;
            m_T.m_x_visible .reserve( 1000 );
            m_T.m_xv_queue  .reserve( 1000 );
            m_T.m_xv_walked .reserve( 1000 );
            m_T.m_xv_walk   .reserve( 1000 );
            m_T.m_sAlloc    .reserve(1000);
            m_T.m_ridges    .reserve( 1000 );
            m_T.m_deref.setBuf( &m_ptStore );

            m_hull_Q.setBit( clarkson93::simplex::ENUMERATE_QUEUED   );
            m_hull_E.setBit( clarkson93::simplex::ENUMERATE_EXPANDED );

            m_hull_Q.reserve(1000);
            m_hull_E.reserve(1000);
        }

        ~Main()
        {
            m_layout.saveValues( m_stateFile );
        }

        void addPoint( double x, double y )
        {
            m_ptStore.push_back( m_inv( Point( x,y,0 ) ) );

            if( m_ptStore.size() < 4 )
                {}
            else if( m_ptStore.size() == 4 )
                m_T.init( 0, 4, [](uint i){return i;} );
            else
            {
                PointRef idx = m_ptStore.size() - 1;
                m_T.insert( idx );
            }

            m_view.queue_draw();
        }

        void run()
        {
            m_gtkmm.run( *m_layout.widget<Gtk::Window>("main") );
        }

        void draw( const Cairo::RefPtr<Cairo::Context>& ctx )
        {
            if(m_ptStore.size() < 4)
                return;

            m_hull_Q.clear();
            m_hull_E.clear();

            // start the stack at some hull simplex
            m_hull_Q.push( m_T.m_hullSimplex );

            // pointer to the point buffer
            Point* pts = m_ptStore.data();

            while( m_hull_Q.size() > 0 )
            {
                // pop a simplex off the stack
                Simplex* S = m_hull_Q.pop();

                // it is a nearest size of the inversion center is not
                // visible, so draw it
                if( !S->isVisible( m_inv.center() ) )
                {
                    for(int i=0; i < 3; i++)
                    {
                        int j = (i+1)%3;
                        int ii = i+1;
                        int jj = j+1;
                        Point x0 = m_inv( pts[S->V[ii]] );
                        Point x1 = m_inv( pts[S->V[jj]] );
                        ctx->move_to( (double)x0[0], (double)x0[1] );
                        ctx->line_to( (double)x1[0], (double)x1[1] );
                    }
                }

                // in any case, expand it
                m_hull_E.push(S);

                // get all it's infinite neighbors
                for(int i=1; i < 4; i++)
                {
                    Simplex* N = S->N[i];
                    // if we haven't already queued or expanded it then queue
                    // it
                    if( !m_hull_Q.isMember(N) && !m_hull_E.isMember(N) )
                        m_hull_Q.push(N);
                }
            }

            ctx->set_source_rgb(0,1,0);
            ctx->set_line_width(0.001);
            ctx->stroke();
        }
};



} // namespace test
} // namespace delaunay
} // namespace mpblocks



namespace ns = mpblocks::delaunay::test;


int main(int argc, char** argv)
{
    ns::Main m_app;
    m_app.run();
    return 0;
}








