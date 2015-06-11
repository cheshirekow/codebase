/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of clarkson93.
 *
 *  clarkson93 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  clarkson93 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with clarkson93.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CLARKSON93_EXAMPLE_TRAITS_H_
#define CLARKSON93_EXAMPLE_TRAITS_H_

namespace clarkson93 {

/// documents the interface for Traits : encapsulates various policies for the
/// Triangulation
/**
 *  @todo   Should these traits be split up into multiple policy classes?
 */
struct ExampleTraits {
  /// optimization levels
  /**
   *      level   | details
   *      ------- | ---------------
   *      0       | Templated Traits implementation following the paper,
   *      1       | simple optimizations that do not change the algorithm
   *      2       |
   *
   */
  static const int OptLevel = 1;

  /// dimension of the embeded space, use clarkson93::Dynamic for a
  /// datastructure whose dimension is set at runtime
  static const unsigned int NDim = 2;

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
  typedef Eigen::Matrix<Scalar, 2, 1> Point;

  /// a reference to a point
  /**
   *  In the example traits this is merely a pointer. This type exists to
   *  allow different methods of pointing to Point data structures.
   *  For instance you may wish to typedef PointRef as an unsigned int, and
   *  let PointRef be an index into an array of points.
   *
   *  @see ExampleTraits::PointDeref
   */
  typedef const Point* PointRef;

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
  struct PointDeref {
    const Point& operator()(PointRef ptr) {
      return *ptr;
    }
  };

  /// we must forward declare Simplex in order to specify that SimplexRef is
  /// a pointer to a simplex
  struct Simplex;

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
  struct Simplex : clarkson93::SimplexBase<ExampleTraits> {
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
  struct SimplexFactory : public std::vector<Simplex> {
    typedef std::vector<Simplex> Base;

    /// construct an object of type T using the next available memory
    // slot and return it's pointer
    Simplex* create() {
      assert(Base::size() < Base::capacity());
      Base::emplace_back();
      return &Base::back();
    }
  };

  /// the triangulation provides some static callbacks for when hull faces
  /// are added or removed. If we wish to do anything special this is where
  /// we can hook into them. If you do not wish to hook into the callbacks
  /// then simply create an empty structure which has empty implementation
  /// for these
  struct Callback {
    void hullFaceAdded(Simplex* S) {}
    void hullFaceRemoved(Simplex* S) {}
  };
};

}  // namespace clarkson93

#endif  // CLARKSON93_EXAMPLE_TRAITS_H_
