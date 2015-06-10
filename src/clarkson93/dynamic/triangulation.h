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
#ifndef MPBLOCKS_CLARKSON93_DYNAMIC_TRIANGULATION_H_
#define MPBLOCKS_CLARKSON93_DYNAMIC_TRIANGULATION_H_

#include <mpblocks/clarkson93.h>
#include <set>
#include <queue>


namespace   mpblocks {
namespace clarkson93 {
namespace    dynamic {


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
class Triangulation
{
    public:
        // Typedefs
        // -----------------------------------------------------------------------
        static const unsigned int NDim   = Traits::NDim;

        typedef unsigned int                uint;
        typedef typename Traits::Scalar     Scalar;
        typedef typename Traits::Point      Point;
        typedef typename Traits::Simplex    Simplex;
        typedef typename Traits::PointRef   PointRef;
        typedef typename Traits::PointDeref PointDeref;
        typedef typename Traits::Setup      Setup;

        typedef Triangulation<Traits>   Triangulation_t;
        typedef Triangulation<Traits>   This_t;

        typedef HorizonRidge<Traits> HorizonRidge_t;
        typedef SimplexBase<Traits>  SimplexBase_t;

        typedef Stack<Simplex*,       SimplexBits> SimplexSet;
        typedef Stack<Simplex*>       SimplexStack;
        typedef Stack<HorizonRidge_t> HorizonSet;

        typedef typename Traits::template Factory<Simplex>         SimplexAlloc;
        typedef typename Traits::template Factory<HorizonRidge_t>  HorizonAlloc;
        typedef std::set<PointRef>                      PointSet;

        // priority queue stuff
        typedef Indexed<Scalar,Simplex*>    PQ_Key;
        typedef P_Queue<PQ_Key>             WalkQueue;

    public:
        // Data Members
        // -----------------------------------------------------------------------
        uint            m_ndim;        ///< dimension
        Simplex*        m_hullSimplex; ///< a simplex in the hull
        Simplex*        m_origin;      ///< origin simplex
        PointRef        m_antiOrigin;  ///< fictitious point
        PointDeref      m_deref;       ///< dereferences a PointRef

        WalkQueue       m_xv_walk;     ///< walk for x-visible search
        SimplexSet      m_xv_walked; ///< set of simplices ever expanded for
                                       ///  the walk

        SimplexSet      m_x_visible;   ///< set of x-visible simplices
        SimplexSet      m_xv_queue;    ///< search queue for x-visible
                                       ///  building

        HorizonSet      m_ridges;      ///< set of horizon ridges

        SimplexAlloc    m_sAlloc;   ///< simplex allocator

    public:
        Triangulation();
        ~Triangulation();

        /// set's up the triangulation by helping it construct it's
        /// allocators and preallocate it's sets
        void setup( Setup& setup );

        /// builds the initial triangulation from the first @p NDim + 1 points
        /// inserted
        template <class Iterator>
        void init(Iterator begin, Iterator end);

        /// insert a new point into the triangulation and update the convex
        /// hull (if necessary)
        /**
         *  @param  x   const ref to the new point to add
         *  @param  S   an x-visible facet, if null (default) will search for one
         */
        void insert(const PointRef x, Simplex* S=0);

        /// destroys all simplex objects that have been generated and clears all
        /// lists/ sets
        void clear();

        /// return the allocator (for debugging)
        SimplexAlloc& getAllocator(){ return m_sAlloc; }

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
        void find_x_visible(PointRef x, Simplex* S=0);

        // update each x-visible simplex by adding the point x as the peak
        // vertex, also create new simplices
        void alter_x_visible(PointRef x);
};



} // namespace dynamic
} // namespace clarkson93
} // namespace mpblocks














#endif // TRIANGULATION_H_
