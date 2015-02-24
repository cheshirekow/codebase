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
 *  @file   /home/josh/Codes/cpp/mpblocks2/convex_hull/include/mpblocks/clarkson93/ExampleTraits.h
 *
 *  @date   Jun 22, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DELAUNAY_TEST_CLARKSONTRAITS_H_
#define MPBLOCKS_DELAUNAY_TEST_CLARKSONTRAITS_H_

#include <mpblocks/clarkson93.h>

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

    /// acts like a uint but has distinct type
    struct ConstPointRef
    {
        uint m_storage;
        ConstPointRef( uint in=0 ): m_storage(in){}
        operator uint&(){ return m_storage; }
    };

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
            Point* m_buf;

        public:
            PointDeref( Point* buf=0 ){ setBuf(buf); }
            void setBuf( Point* buf ) { m_buf = buf; }
                  Point& operator()(      PointRef idx ){ return m_buf[idx]; }
            const Point& operator()( ConstPointRef idx ){ return m_buf[idx]; }
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
    template <typename T>
    struct Factory
    {
        T*      m_start;    ///< start of our allocated buffer
        T*      m_next;     ///< next unused block of memory
        T*      m_end;      ///< past-the-end iterator

        /// initializes pointers to null
        Factory():
            m_start(0),
            m_end(0),
            m_next(0)
        {}

        /// frees any memory
        ~Factory()
        {
            dealloc();
        }

        /// frees allocated memory if we have allocated any
        void dealloc()
        {
            if(m_start)
            {
                ::operator delete( m_start );
                m_start = 0;
                m_end   = 0;
                m_next  = 0;
            }
        }

        /// allocates a block of memory large enough for n objects of type T
        void alloc( size_t n )
        {
            dealloc();
            if( n > 0 )
            {
                m_start = (Simplex*)::operator new( n*sizeof(T) );
                m_next  = m_start;
                m_end   = m_start + n;
            }
        }

        /// destroy all of the objects we have created and reset the memory
        /// pointer to the beginning of the buffer
        void clear()
        {
            while(m_next > m_start )
            {
                --m_next;
                m_next->~T();
            }
        }

        /// construct an object of type T using the next available memory
        // slot and return it's pointer
        template < typename ...P >
        T* create( P... params )
        {
            assert( m_next != m_end );
            T* next = m_next++;
            return new(next) T( params... );
        }

        /// construct an object of type T using the next available memory
        // slot and return it's pointer
        T* create()
        {
            assert( m_next != m_end );
            T* next = m_next++;
            return new(next) T( );
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

    /// the triangulation takes a reference to one of these objects in it's
    /// setup() method.
    struct Setup
    {
        Point* m_ptBuf;
        typedef unsigned int uint;

        Setup( Point* buf ): m_ptBuf(buf){}

        /// this is only required if NDim = Dynamic (-1), in which case
        /// this is how we tell the triangulation what the dimension is
        /// now
        uint nDim(){ return NDim; }

        /// size of the hull set enumerator to preallocate
        uint hullPrealloc(){ return 500; }

        /// size of the horizion set to preallocate
        uint horizonPrealloc(){ return 500; }

        /// size of x_visible to preallocate
        uint xVisiblePrealloc(){ return 1000; }

        /// returns a special PointRef which is used as the anti origin.
        PointRef antiOrigin(){ return -1; }

        /// returns an object that can dereference a PointRef
        PointDeref deref(){ return PointDeref(m_ptBuf); }

        /// returns a pointer to a callback object
        Callback callback(){ return Callback(); }

        /// sets up the allocator however we need it to
        void setupAlloc( Factory<Simplex>& alloc )
            { alloc.alloc(1000); }

    };


};

} // namespace test
} // namespace delaunay
} // namespace mpblocks















#endif // EXAMPLETRAITS_H_
