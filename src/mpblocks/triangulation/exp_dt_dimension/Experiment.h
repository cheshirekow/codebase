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
 *  @file   /home/josh/Codes/cpp/mpblocks2/triangulation/src/exp_dt_dimension/Experiment.h
 *
 *  @date   Aug 8, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_EXP_DT_DIMENSION_EXPERIMENT_H_
#define MPBLOCKS_EXP_DT_DIMENSION_EXPERIMENT_H_

#include "ExperimentBase.h"

#include <mpblocks/brown79.hpp>
#include <mpblocks/clarkson93.hpp>
#include <mpblocks/btps.h>
#include <mpblocks/utility/Timespec.h>
#include <cpp_pthreads.h>

#include <functional>
#include <bitset>

namespace         mpblocks {
namespace exp_dt_dimension {




/// an example traits structure from which a balacned tree of partial sums
/// may be instantiated
struct BTPSTraits
{
    /// need to forward declare Node so we can make NodeRef be a pointer
    struct Node;

    /// some type which stores uniquely identifies nodes, for instance
    /// a node pointer or an index into an array
    typedef Node* NodeRef;

    /// our node type extends the basic node type
    struct Node:
        btps::BasicNode<BTPSTraits>
    {
        Node( double weight = 0 ):
            BasicNode<BTPSTraits>(weight)
        {}

    };

    /// a callable type which implements the primitives required to access
    /// fields of a node
    struct NodeOps
    {
        /// you can leave this one off you if you dont need removal
        NodeRef&   parent( NodeRef N ){ return N->parent; }

        /// return the left child of N
        NodeRef&     left( NodeRef N ){ return N->left;   }

        /// return the right child of N
        NodeRef&    right( NodeRef N ){ return N->right;  }

        /// return the weight of this node in particular
        double     weight( NodeRef N ){ return N->weight; }

        /// return the cumulative weight of the subtree, the return type
        /// is deduced by the tree template and can be anything modeling a
        /// real number
        double&       cum( NodeRef N ){ return N->cumweight; }

        /// return the subtree node count
        uint32_t&   count( NodeRef N ){ return N->count;  }
    };
};


typedef BTPSTraits::Node        BTPSNode;
typedef btps::Tree<BTPSTraits>  BTPSTree;



/// documents the interface for Traits : encapsulates various policies for the
/// Triangulation
template <typename SCALAR, int NDIM>
struct ClarksonTraits
{
    typedef unsigned int uint;

    /// dimension of the embeded space, use clarkson93::Dynamic for a
    /// datastructure whose dimension is set at runtime
    static const unsigned int NDim = (NDIM+1);

    /// optimization level
    static const unsigned int OptLevel = 0;

    /// number format for scalar numbers
    typedef SCALAR Scalar;

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
    typedef Eigen::Matrix<Scalar, NDim, 1> Point;

    // forward declare so we can define SimplexRef
    struct Simplex;

    /// a reference to a point
    typedef uint PointRef;

    /// a reference to a simplex
    typedef Simplex* SimplexRef;

    /// proves a means of turning a PointRef into a Point&
    /**
     *  In the example traits PointRef is a Point* so we simply dereference this
     *  pointer. If PointRef were an index into an array of Point structures,
     *  then Deref should store a pointer to the beginning of that array.
     */
    class Deref
    {
        private:
            std::vector<Point>* m_buf;

        public:
            Deref( std::vector<Point>* buf=0 ){ setBuf(buf); }
            void setBuf( std::vector<Point>* buf ) { m_buf = buf; }

            Point&      point(  PointRef idx ){ return (*m_buf)[idx]; }
            Simplex& simplex( SimplexRef ptr ){ return *ptr; }
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
            clarkson93::Simplex2<ClarksonTraits>,
            BTPSNode
    {
        typedef clarkson93::Simplex2<ClarksonTraits> Base;

        Eigen::Matrix<SCALAR,NDIM,1>    c_c;    ///< circum center
        SCALAR                          c_r;    ///< circum radius
        bool                            c_has;  ///< has center

        Simplex():
            Base(-1,0),
            c_r(0),
            c_has(false)
        {}
    };

    /// Triangulation will derive from this in order to inherit operations
    /// for simplices
    typedef clarkson93::SimplexOps< ClarksonTraits > SimplexOps;

    /// template for allocators & memory managers
    /**
     *  The triangulation doesn't support removal of points. Thus simplices are
     *  never deleted individually (rather, all at once when the triangulation
     *  is cleared). Thus we divert memory management to the user of the
     *  libary. The allocator is responsible for keeping track of every object
     *  it allocates an be delete them all when cleared
     *
     *  This example preallocates a fixed size array of storage for objects.
     *  When create() is called it takes the next unused pointers, calls
     *  in-place operator new() and then returns the pointer. When it is
     *  reset, it goes through the list of allocated pointers and calls the
     *  destructor (ptr)->~T() on each of them.
     *
     *  For the most part, Alloc<T> is only used to construct POD types so
     *  this is a bit of a moot point.
     *
     *  @note: Alloc must be default constructable. It will be passed to
     *         setup.prepare( Alloc<T> ) so if you need to do any allocator
     *         initialization you must do it in Setup
     */
    struct SimplexMgr:
        public std::vector<Simplex>
    {
        typedef std::vector<Simplex> base_t;

        /// construct an object of type T using the next available memory
        // slot and return it's pointer
        SimplexRef create()
        {
            assert( base_t::size() < base_t::capacity() );
            base_t::emplace_back( );
            return &base_t::back();
        }
    };

    /// the triangulation provides some static callbacks for when hull faces
    /// are added or removed. If we wish to do anything special this is where
    /// we can hook into them. If you do not wish to hook into the callbacks
    /// then simply create an empty structure which has empty implementation
    /// for these
    struct Callback
    {
        std::function<void(Simplex*)> sig_hullFaceAdded;
        std::function<void(Simplex*)> sig_hullFaceRemoved;

        void hullFaceAdded(Simplex* S)
        {
            sig_hullFaceAdded(S);
        }

        void hullFaceRemoved(Simplex* S)
        {
            sig_hullFaceRemoved(S);
        }
    };

};



template <typename T>
constexpr T factoral( T n )
{
    return n == 1 ? 1 : n*factoral(n-1);
}



template <typename Scalar>
struct TypeToString{ static constexpr const char* Value = "UNDEFINED"; };

template <> struct TypeToString<float>
{
    static constexpr const char* Value = "float";
};

template <> struct TypeToString<double>
{
    static constexpr const char* Value = "double";
};

template <bool SmartSample>
struct StrategyToString
{
    static constexpr const char* Value = "smart_sample";
};

template <>
struct StrategyToString<false>
{
    static constexpr const char* Value = "direct_sample";
};



template <typename Scalar, int NDim, bool SmartSample>
class Experiment:
    public ExperimentBase
{
    public:
        typedef Experiment<Scalar,NDim,SmartSample> This;

        typedef ClarksonTraits<Scalar,NDim> CTraits;
        typedef typename CTraits::Simplex     Simplex;
        typedef typename CTraits::Point       Point;
        typedef typename CTraits::PointRef    PointRef;
        typedef typename CTraits::Deref       Deref;

        typedef clarkson93::Triangulation<CTraits>      Triangulation;
        typedef brown79::Inversion<CTraits>             Inversion;

        static const     int    s_nFactoral = factoral(NDim);
        static constexpr Scalar s_bounds    = 10.0;

    private:
        Inversion           m_inv;
        Deref               m_deref;
        std::vector<Point>  m_ptStore;
        std::vector<Point>  m_ptStoreN;

        Triangulation   m_T;

        BTPSTree        m_btps;
        BTPSNode        m_btpsNil;

        int m_newSimplices;     ///< number of clarkson simplices created
        int m_hullSimplices;    ///< number of added/removed hull simplices

    public:
        Experiment():
            m_deref(&m_ptStore),
            m_btps(&m_btpsNil),
            m_newSimplices(0),
            m_hullSimplices(0)
        {
            // we want the center to be a little bit off-center of the
            // workspace or else the initial triangulation will be co-planar
            // after inversion
            Point invCenter;
            invCenter.fill(3*s_bounds/4);
            invCenter[NDim] = s_bounds;

            m_inv.init( invCenter, s_bounds );
            m_T.m_deref.setBuf( &m_ptStore );

            using namespace std::placeholders;
            m_T.m_callback.sig_hullFaceAdded =
                    std::bind( &This::simplexAdded, this, _1);
            m_T.m_callback.sig_hullFaceRemoved =
                    std::bind( &This::simplexRemoved, this, _1);

            m_T.m_antiOrigin = -1;

            std::stringstream filename;
            filename << "data_" << TypeToString<Scalar>::Value
                     << "_" << StrategyToString<SmartSample>::Value
                     << "_" << NDim
                     << ".csv";
            m_filename = filename.str();
        }

        virtual ~Experiment(){}

        void initExperiment(int nPoints, int nSimplices)
        {
            ExperimentBase::initExperiment(nPoints);
            m_T.clear();
            m_btps.clear();
            m_ptStore.clear();  ///< clear so realloc doesn't copy
            m_ptStore.reserve(nPoints);
            m_ptStoreN.clear();
            m_ptStoreN.reserve(nPoints);
            m_T.m_xv_queue  .reserve( nSimplices );
            m_T.m_xv_walked .reserve( nSimplices );
            m_T.m_sMgr      .reserve( nSimplices );
            m_T.m_ridges    .reserve( nSimplices );

        }

        void computeMeasure( Simplex* S )
        {
            typedef Eigen::Matrix<Scalar,NDim,NDim> Matrix;
            typedef Eigen::Matrix<Scalar,NDim,1>    Vector;

            PointRef peak = S->V[S->iPeak];
            std::vector<PointRef> pRefs;
            pRefs.reserve(NDim+2);
            std::copy_if( S->V, S->V + NDim+2,
                            std::back_inserter(pRefs),
                            [peak](PointRef v){ return peak != v; } );

            Matrix A;
            Vector x0 = m_inv( m_deref.point(pRefs[0]) ).block(0,0,NDim,1);
            for(int i=1; i < NDim+1; i++)
            {
                Vector xi   = m_inv( m_deref.point(pRefs[i]) ).block(0,0,NDim,1);
                A.col(i-1)  = (xi - x0);
            }

            S->weight = std::abs( A.lu().determinant() / s_nFactoral );
        }

        void computeCenter( Simplex* S )
        {
            typedef Eigen::Matrix<Scalar,NDim,NDim> Matrix;
            typedef Eigen::Matrix<Scalar,NDim,1>    Vector;

            // calculate circumcenter of the first simplex
            // see 2012-10-26-Note-11-41_circumcenter.xoj for math
            Matrix   A;
            Vector   b;

            PointRef peak = S->V[S->iPeak];
            std::vector<PointRef> pRefs;
            pRefs.reserve(NDim+2);
            std::copy_if( S->V, S->V + NDim+2,
                            std::back_inserter(pRefs),
                            [peak](PointRef v){ return peak != v; } );

            Vector x0 = m_inv( m_deref.point(pRefs[0]) ).block(0,0,2,1);
            for(int i=1; i < NDim+2; i++)
            {
                Vector xi   = m_inv( m_deref.point(pRefs[i]) ).block(0,0,2,1);
                Vector dv   = 2*(xi - x0);
                A.row(i-1)  = dv;
                b[i-1]      = xi.squaredNorm() - x0.squaredNorm();
            }

            // the circum center
            S->c_c  = A.fullPivLu().solve(b);

            // squared radius of the circumsphere
            S->c_r = ( S->c_c - x0 ).norm();
            S->c_has = true;
        }

        void simplexAdded( Simplex* S )
        {
            S->parent = 0;
            S->right  = 0;
            S->left   = 0;

            m_newSimplices++;
            if( m_T.isVisible(*S,m_inv.center() ) )
                return;

            m_hullSimplices++;
            if( SmartSample )
            {
                computeMeasure(S);
                m_btps.insert(S);
            }
        }

        void simplexRemoved( Simplex* S )
        {
            if( m_T.isVisible(*S,m_inv.center() ) )
                return;

            m_hullSimplices--;

            if( SmartSample )
            {
                m_btps.remove( S );
                S->parent = 0;
                S->right  = 0;
                S->left   = 0;
            }
        }

        void initialTriangulation()
        {
            m_ptStore.clear();
            m_ptStoreN.clear();
            m_T.clear();
            m_btps.clear();

            Point point;
            point[NDim] =0;

            boost::uniform_real<Scalar> u(0,1e-3);

            // attempt to break symetries with an interior point
            point.fill( s_bounds / 2 );
            point[NDim] = 0;
            m_ptStoreN.push_back( point );
            m_ptStore.push_back( m_inv(point) );

            for( unsigned long i=0; i < (0x01 << NDim); i++)
            {
                std::bitset<NDim> bits(i);
                for( int j=0; j < NDim; j++)
                    point[j] = u(m_rng) + ( bits[j] ? 0 : s_bounds );
                m_ptStoreN.push_back( point );
                m_ptStore.push_back( m_inv(point) );

                if( m_ptStore.size() == NDim+2 )
                {
//                    std::cout << "Building initial triangulation\n";
//                    for( auto point : m_ptStore )
//                        std::cout << "   " << point.transpose() << "\n";
                    m_T.init( 0, NDim+2, [](unsigned int k){return k;} );
                }
                else if( m_ptStore.size() > NDim+2 )
                {
//                    std::cout << "inserting " << m_ptStore.back().transpose() << "\n";
                    m_T.insert(m_ptStore.size()-1);
                }
            }
        }

        void step()
        {
            utility::Timespec start,finish;

            m_newSimplices  = 0;
            m_hullSimplices = 0;

            Point       newPoint;
            newPoint[NDim] = 0;

            clock_gettime( CLOCK_MONOTONIC, &start );

            if( SmartSample )
            {
                boost::uniform_real<Scalar> u(0,1);

                // select a node from the weighted disribution
                BTPSNode* node = m_btps.findInterval( u(m_rng) );

                // cast it to a simplex
                Simplex* S = static_cast<Simplex*>(node);
                assert( m_T.isInfinite( *S, m_T.m_antiOrigin ) );

                // generate a random barycentric coordinate
                std::set<double> Z;
                for(int i=0; i < NDim; i++)
                    Z.insert( u(m_rng) );
                Z.insert(1);

                std::vector<double> lambda;
                lambda.reserve(NDim+1);
                double z_prev = 0;
                for( double z : Z )
                {
                    lambda.push_back( z - z_prev );
                    z_prev = z;
                }

                // build a list of non-peak vertices
                PointRef peak = S->V[S->iPeak];
                std::vector<PointRef> pRefs;
                pRefs.reserve(NDim+2);
                std::copy_if( S->V, S->V + NDim+2,
                                std::back_inserter(pRefs),
                                [peak](PointRef v){ return peak != v; } );

                // compute the point at that coordinate
                newPoint.fill(0);
                for(int i=0; i < NDim+1; i++)
                {
                    assert( pRefs[i] < m_ptStoreN.size() );
                    newPoint = newPoint
                        + lambda[i] * m_ptStoreN[ pRefs[i] ];
                }
                newPoint[NDim] = 0;

                if( !m_T.isVisible( *S, m_inv(newPoint) ) )
                {
                    std::cout << newPoint.transpose() << "\n    is not inside: \n";
                    for( int i=0; i < NDim+1; i++ )
                        std::cout << "   " << m_ptStoreN[pRefs[i]].transpose() << "\n";
                }

                assert( m_T.isVisible( *S, m_inv(newPoint) ) );

                // insert that point into the triangulation
                m_ptStoreN.push_back( newPoint );
                m_ptStore.push_back( m_inv(newPoint) );
                m_T.insert(m_ptStore.size()-1,S);
            }
            else
            {
                boost::uniform_real<Scalar> u(0,s_bounds);
                for(int i=0; i < NDim; i++)
                    newPoint[i] = u(m_rng);

                // insert that point into the triangulation
                m_ptStoreN.push_back( newPoint  );
                m_ptStore.push_back( m_inv(newPoint) );
                m_T.insert(m_ptStore.size()-1);
            }



            clock_gettime( CLOCK_MONOTONIC, &finish );



            updateClarksonCount( m_newSimplices );
            updateHullCount(    m_hullSimplices );
            appendTime( (finish-start).tv_nsec  );
        }



};


} //< namespace exp_dt_dimension
} //< namespace mpblocks




#endif // EXPERIMENT_H_
