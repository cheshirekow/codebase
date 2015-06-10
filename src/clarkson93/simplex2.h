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
#ifndef MPBLOCKS_CLARKSON93_SIMPLEX2_H_
#define MPBLOCKS_CLARKSON93_SIMPLEX2_H_

#include <mpblocks/clarkson93.h>

namespace   mpblocks {
namespace clarkson93 {



/// A simplex is the convex hull of d+1 points in general position (IGP), i.e.
/// they do not lie on the same hyperplane
/**
 *  In order to implicitly maintain the convensions of Clarkson.compgeo93,
 *  we will order the vertices and facets of the simplex such that the i'th
 *  facet is opposite the i'th vertex (using the definition of clarkson), and
 *  thus is defined by the d vertices
 *  excluding the i'th. In addition, the i'th neighbor is the neighbor sharing
 *  the i'th facet and opposite the i'th vertex.
 *
 *  In this second version of the Simplex structure we do not maintain the
 *  zeroth vertex/Neighbor pair as the designated peak vertex, base facet.
 *  Instead we store the vertex references in sorted order so that we can
 *  efficiently perform queries regarding various dimensional facets of
 *  simplex.
 *
 *  The simplex also stores a normal vector \p n and an offset \p c defining
 *  a linear inequality whose hyperplane is coincident to the 'base' facet and
 *  oriented such that n' x < c is the half-space containing the 'peak' vertex
 */
template <class Traits>
struct Simplex2
{
    // Typedefs
    // -----------------------------------------------------------------------
    static const unsigned int NDim = Traits::NDim;
    typedef typename Traits::Scalar     Scalar;
    typedef typename Traits::Point      Point;
    typedef typename Traits::SimplexRef SimplexRef;
    typedef typename Traits::PointRef   PointRef;
    typedef typename Traits::Deref      Deref;
    typedef std::bitset<simplex::NUM_BITS>  BitSet;

    // Data Members
    // -----------------------------------------------------------------------
    uint32_t    iPeak;       ///< index of the peak vertex
    PointRef    V[NDim+1];   ///< vertices of the simplex
    SimplexRef  N[NDim+1];   ///< simplices which share a facet
    BitSet      sets;        ///< sets this simplex is a member of

    Point   n;  ///< normal vector of base facet
    Scalar  o;  ///< offset of base facet inequality hyperplane

    Simplex2( PointRef pNull, SimplexRef sNull ):
        iPeak(0),
        o(0)
    {
        for(int i=0; i < NDim+1; i++)
        {
            V[i] = pNull;
            N[i] = sNull;
        }
        sets.reset();
        n.fill(0);
    }
};



/// a subset of the vertices of a simplex
template <class Traits>
struct Facet2:
    std::vector< typename Traits::PointRef >
{};

/// an interator over neighbors in a simplex
template <class Traits>
struct NeighborIterator
{
    // Typedefs
    // -----------------------------------------------------------------------
    static const unsigned int NDim = Traits::NDim;
    typedef typename Traits::SimplexRef SimplexRef;
    typedef NeighborIterator<Traits>    This;
    typedef Simplex2<Traits>            Simplex;


    Simplex&   m_S;
    uint32_t   m_nIdx;

    NeighborIterator( Simplex& S, uint32_t nIdx ):
        m_S(S),
        m_nIdx(nIdx)
    {}

    SimplexRef  operator*() { return m_S.N[m_nIdx];   }

    This&   operator++()
    {
        m_nIdx++;
        return *this;
    }

    bool operator!=( NeighborIterator<Traits>& other )
    {
        return m_nIdx != other.m_nIdx;
    }
};

/// provides an iterator range for neighbors of a simplex
template <class Traits>
struct NeighborRange
{
    // Typedefs
    // -----------------------------------------------------------------------
    static const unsigned int NDim = Traits::NDim;
    typedef typename Traits::SimplexRef SimplexRef;
    typedef NeighborIterator<Traits>    Iterator;
    typedef Simplex2<Traits>            Simplex;
    typedef Facet2<Traits>              Facet;

    Simplex&   m_S;

    NeighborRange( Simplex& S ):m_S(S){}
    Iterator begin(){ return Iterator(m_S,0);      }
    Iterator end()  { return Iterator(m_S,NDim+1); }
};



/// an interator over vertices in a simplex
template <class Traits>
struct VertexIterator
{
    // Typedefs
    // -----------------------------------------------------------------------
    static const unsigned int NDim = Traits::NDim;
    typedef typename Traits::PointRef   PointRef;
    typedef VertexIterator<Traits>      This;
    typedef Simplex2<Traits>            Simplex;

    Simplex&   m_S;
    uint32_t   m_nIdx;

    VertexIterator( Simplex& S, uint32_t nIdx ):
        m_S(S),
        m_nIdx(nIdx)
    {}

    PointRef  operator*() { return m_S.V[m_nIdx];   }

    This&   operator++()
    {
        m_nIdx++;
        return *this;
    }

    bool operator!=( VertexIterator<Traits>& other )
    {
        return m_nIdx != other.m_nIdx;
    }
};

/// provides an iterator range for vertices of a simplex
template <class Traits>
struct VertexRange
{
    // Typedefs
    // -----------------------------------------------------------------------
    static const unsigned int NDim = Traits::NDim;
    typedef typename Traits::PointRef   PointRef;
    typedef VertexIterator<Traits>      Iterator;
    typedef Simplex2<Traits>            Simplex;

    Simplex&   m_S;

    VertexRange( Simplex& S ):m_S(S){}
    Iterator begin(){ return Iterator(m_S,0);      }
    Iterator end()  { return Iterator(m_S,NDim+1); }
};


/// surrogate for a certain value used in setters
template <typename T>
struct Surrogate
{
    public:
        T& ref;

    Surrogate( T& ref ):ref(ref){}

    template <typename U>
    Surrogate<T>& operator=( U val )
    {
        ref = val;
        return *this;
    }
};

/// implements simplex operations for simplex references, using other
/// references can be easily extended by building on top of this class
template <class Traits>
struct SimplexOps
{
    // Typedefs
    // -----------------------------------------------------------------------
    static const unsigned int NDim = Traits::NDim;
    typedef typename Traits::Scalar     Scalar;
    typedef typename Traits::Point      Point;
    typedef typename Traits::SimplexRef SimplexRef;
    typedef typename Traits::PointRef   PointRef;
    typedef typename Traits::Deref      Deref;
    typedef NeighborRange<Traits>       NRange;
    typedef VertexRange<Traits>         VRange;
    typedef Simplex2<Traits>            Simplex;
    typedef std::bitset<simplex::NUM_BITS>  BitSet;
    typedef BitSet::reference           BitRef;

    // Field accessors
    // -----------------------------------------------------------------------
    PointRef            vertex( Simplex& S, uint32_t i );         ///< vertex at i
    SimplexRef        neighbor( Simplex& S, uint32_t i );         ///< neighbor at i
    SimplexRef  neighborAcross( Simplex& S, PointRef v );         ///< neighbor across v
    SimplexRef    peakNeighbor( Simplex& S );                     ///< neighbor across peak
    PointRef              peak( Simplex& S );                     ///< designated peak vertex
    bool              isMember( Simplex& S, SimplexBits bit );    ///< S is a member of a set
    uint32_t           indexOf( Simplex& S, PointRef v );         ///< index of v in S
    NRange        neighborhood( Simplex& S );                     ///< complete neighborhood
    VRange            vertices( Simplex& S );                     ///< complete vertex set

    /// simultaneously construct the set intersection and the set symmetric
    /// difference of Sa and Sb
    template <typename Output1, typename Output2, typename Output3>
    void  vsetSplit( Simplex& Sa, Simplex& Sb,
                        Output1 a_only, Output2 b_only, Output3 intersect );

    /// builds a list of vertices common to both simplices
    template <typename Output>
    void  vsetIntersection( Simplex& Sa, Simplex& Sb, Output intersect );

    /// builds a list of neighbors that share a facet... this is the set of
    /// neighbors who are accross from a vertex v where v is not in the
    /// facet set
    template <typename Input, typename Output>
    void neighborSharing( Simplex& S, Input first, Input last, Output iter );


    // setters
    // -----------------------------------------------------------------------
    Surrogate<PointRef>             setPeak( Simplex& S );
    Surrogate<PointRef>           setVertex( Simplex& S, uint32_t i );    ///< set vertex i
    Surrogate<SimplexRef>       setNeighbor( Simplex& S, uint32_t i );    ///< set neighbor i
    Surrogate<SimplexRef> setNeighborAcross( Simplex& S, PointRef v );    ///< set neighbor across v
    BitRef                        setMember( Simplex& S, SimplexBits bit );   ///< set S's membership in bit

    // Mutators
    // -----------------------------------------------------------------------
    /// after filling vertices, lay them out in sorted order
    void finish( Simplex& );

    /// make two simplices neighbors across a common facet

    /// compute the base facet normal and offset
    void computeBase( Simplex& S, Deref& deref );

    /// orient the base facete normal by ensuring that the point x
    /// lies on the appropriate half-space
    /// @f$ n \cdot x \le c @f$ )
    void orientBase( Simplex& S, const Point& x, Orientation orient=INSIDE  );

    // point queries
    // -----------------------------------------------------------------------

    /// returns the distance of x to the base facet, used in walking the
    /// triangulation to x
    Scalar normalProjection( const Simplex& S, const Point& x );

    /// returns true if the base vertex is the anti origin
    bool isInfinite( const Simplex& S, PointRef antiOrigin );

    /// returns true if x is on the inside of the base facet (i.e. x is in the
    /// same half space as the simplex)
    bool isVisible( const Simplex& S, const Point& x );
};






} // namespace clarkson93
} // namespace mpblocks















#endif // SIMPLEX_H_
