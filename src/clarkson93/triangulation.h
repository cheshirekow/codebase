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
#ifndef MPBLOCKS_CLARKSON93_TRIANGULATION_H_
#define MPBLOCKS_CLARKSON93_TRIANGULATION_H_

#include <mpblocks/clarkson93.h>
#include <set>
#include <queue>


namespace   mpblocks {
namespace clarkson93 {


/// simply maps a complile-time integer to a type
template <unsigned int>
struct OptLevel{};

/// Template meta function evaluating whether or not a traits class specifies
/// an optimization level.
/**
 *  Usage:
 *  @code
      const bool ops_defined = OptDefined<SomeType>::Value;
@endcode
 */
template<typename T> struct OptDefined
{
    template<typename C> static char  test( decltype(C::OptLevel) );
    template<typename C> static char* test(...);
    enum{ Value = sizeof(test<T>(0))==1 };
};

/// Template meta function evaluating the optimization level as specified by
/// the traits class
/**
 *  Defines an internal value which evaluates to the optimization level.
 *  Usage:
 *  @code
     enum{ OptLevel = OptLevelGet<SomeType>::Value };
@endcode
 */
template<typename T,bool Defined=OptDefined<T>::Value > struct OptLevelGet
{
    enum{ Value = T::OptLevel };
};

template<typename T> struct OptLevelGet<T,false>
{
    enum{ Value = 0 };
};






/// misleadingly-named data structure, is actually a "simplexification", the
/// dimension agnostic analog of a triangulation
/**
 *  By the convensions of Clarkson.compgeo93, the first (i.e. 0 index) simplex
 *  is the "origin" simplex. The 'origin' (@f$ O @f$) is a point inside that simplex
 *  (we'll pick the geometric mean of the vertices). The 'anti-origin'
 *  (@f$ \overbar{o} @f$) is a fictitous vertex which is a member of all
 *  simplices
 *  on the convex hull. It is analogous to CGAL's 'vertex at infinity'. We'll
 *  give it a value of @f$ [0 0 0...] @f$, but it's address is used to identify
 *  it as a special point, and it's value is never used in calculations.
 */
template < class Traits >
class Triangulation:
        public Traits::SimplexOps
{
    public:
        // Typedefs
        // -----------------------------------------------------------------------
        static const unsigned int NDim          = Traits::NDim;
        static const OptLevel< OptLevelGet<Traits>::Value >  s_optLvl;

        typedef typename Traits::Scalar     Scalar;
        typedef typename Traits::Point      Point;
        typedef typename Traits::Simplex    Simplex;
        typedef typename Traits::PointRef   PointRef;
        typedef typename Traits::SimplexRef SimplexRef;
        typedef typename Traits::Deref      Deref;
        typedef typename Traits::Callback   Callback;
        typedef typename Traits::SimplexMgr SimplexMgr;

        typedef Triangulation<Traits>   This;
        typedef HorizonRidge<Traits>    Ridge;

        typedef std::set<PointRef>      PointSet;
        typedef std::vector<SimplexRef> SimplexSet;
        typedef std::vector<Ridge>      HorizonSet;

        // priority queue stuff
        typedef Indexed<Scalar,SimplexRef>  PQ_Key;
        typedef P_Queue<PQ_Key>             WalkQueue;

    public:
        // Data Members
        // -----------------------------------------------------------------------
        SimplexRef      m_hullSimplex; ///< a simplex in the hull
        SimplexRef      m_origin;      ///< origin simplex
        PointRef        m_antiOrigin;  ///< fictitious point
        Deref           m_deref;       ///< dereferences a PointRef or SimplexRef

        WalkQueue       m_xv_queue;    ///< walk for x-visible search
        SimplexSet      m_xv_walked;   ///< set of simplices ever expanded for
                                       ///  the walk

        SimplexSet      m_xvh;         ///< set of x-visible hull simplices
        SimplexSet      m_xvh_queue;   ///< search queue for x-visible hull

        HorizonSet      m_ridges;      ///< set of horizon ridges

        SimplexMgr      m_sMgr;        ///< simplex manager
        Callback        m_callback;    ///< event hooks

    public:
        Triangulation();
        ~Triangulation();

        /// inherited from SimplexOps
        using Traits::SimplexOps::vertex;
        using Traits::SimplexOps::neighbor;
        using Traits::SimplexOps::neighborAcross;
        using Traits::SimplexOps::peakNeighbor;
        using Traits::SimplexOps::peak;
        using Traits::SimplexOps::isMember;
        using Traits::SimplexOps::indexOf;
        using Traits::SimplexOps::neighborhood;
        using Traits::SimplexOps::vsetSplit;
        using Traits::SimplexOps::vsetIntersection;
        using Traits::SimplexOps::neighborSharing;
        using Traits::SimplexOps::setPeak;
        using Traits::SimplexOps::setVertex;
        using Traits::SimplexOps::setNeighbor;
        using Traits::SimplexOps::setNeighborAcross;
        using Traits::SimplexOps::setMember;
        using Traits::SimplexOps::finish;
        using Traits::SimplexOps::computeBase;
        using Traits::SimplexOps::orientBase;
        using Traits::SimplexOps::normalProjection;
        using Traits::SimplexOps::isInfinite;
        using Traits::SimplexOps::isVisible;


        /// builds the initial triangulation from the first @p NDim + 1 points
        /// inserted
        template <class Iterator, class Deiter>
        void init(Iterator begin, Iterator end, Deiter deiter);

        /// insert a new point into the triangulation and update the convex
        /// hull (if necessary)
        /**
         *  @param  x   const ref to the new point to add
         *  @param  S   an x-visible facet, if null (default) will search for one
         */
        void insert(const PointRef x, SimplexRef S);
        void insert(const PointRef x);

        /// destroys all simplex objects that have been generated and clears all
        /// lists/ sets
        void clear();

        /// find the set of @f$ x\mathrm{-visible} @f$ facets and put them in
        /// @p x_visible
        /**
         *  Using the first method of Clarkson.compgeo93, we walk along the segment
         *  @f$ \overbar{Ox} @f$, starting at @f$ O @f$ . If this walk enters a
         *  simplex whose peak vertex is the anti-origin, then
         *  an x-visible current facet has been found. Otherwise a simplex of
         *  T containing @f$ x @f$ has been found, showing that
         *  @f$ x \in \mathrm{hull} R @f$
         */
        Simplex* find_x_visible(PointRef x, SimplexRef S);

        /// given a simplex S which is x-visible and infinite, fill the set of
        /// all x-visible and infinite facets
        void fill_x_visible( const OptLevel<0>&, PointRef x, SimplexRef S);

        // update each x-visible simplex by adding the point x as the peak
        // vertex, also create new simplices
        void alter_x_visible( const OptLevel<0>&, PointRef x);
};


template < class Traits >
const OptLevel< OptLevelGet<Traits>::Value >
    Triangulation<Traits>::s_optLvl = OptLevel< OptLevelGet<Traits>::Value >();



} // namespace clarkson93
} // namespace mpblocks














#endif // TRIANGULATION_H_
