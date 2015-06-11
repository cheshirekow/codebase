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
#ifndef CLARKSON93_EXAMPLE_TRAITS2_H_
#define CLARKSON93_EXAMPLE_TRAITS2_H_

namespace clarkson93 {

/// documents the interface for Traits : encapsulates various policies for the
/// Triangulation
/**
 *  @todo   Should these traits be split up into multiple policy classes?
 */
template <typename SCALAR, unsigned int NDIM>
struct ExampleTraits2 {
  typedef ExampleTraits2<SCALAR, NDIM> This;

  /// optimization levels
  /**
   *      level   | details
   *      ------- | ---------------
   *      0       | General implementation wigh generic point and simplex
   *              | references, and generic operations for each
   */
  static const int OptLevel = 0;

  /// dimension of the embeded space, use clarkson93::Dynamic for a
  /// datastructure whose dimension is set at runtime
  static const unsigned int NDim = NDIM;

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
  typedef Point* PointRef;

  /// a reference to a simplex
  typedef Simplex* SimplexRef;

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
  struct Deref {
    Point& point(PointRef ptr) { return *ptr; }
    Simplex& simplex(SimplexRef ptr) { return *ptr; }
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
  struct Simplex : clarkson93::Simplex2<This> {
    typedef clarkson93::Simplex2<This> base_t;
    Simplex() : base_t(0, 0) {}
  };

  /// Triangulation will derive from this in order to inherit operations
  /// for simplices
  typedef clarkson93::SimplexOps<This> SimplexOps;

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
  struct SimplexMgr : public std::vector<Simplex> {
    typedef std::vector<Simplex> base_t;

    /// construct an object of type T using the next available memory
    // slot and return it's pointer
    SimplexRef create() {
      assert(base_t::size() < base_t::capacity());
      base_t::emplace_back();
      return &base_t::back();
    }
  };

  /// the triangulation provides some static callbacks for when hull faces
  /// are added or removed. If we wish to do anything special this is where
  /// we can hook into them. If you do not wish to hook into the callbacks
  /// then simply create an empty structure which has empty implementation
  /// for these
  struct Callback {
    void hullFaceAdded(SimplexRef S) {}
    void hullFaceRemoved(SimplexRef S) {}
  };
};

}  // namespace clarkson93

#endif  // CLARKSON93_EXAMPLE_TRAITS2_H_
