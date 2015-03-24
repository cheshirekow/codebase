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
 *  @file
 *  @date   Jul 11, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  provides example traits for instantiating an edelsbrunner
 *          triangulation object
 */

#ifndef EDELSBRUNNER96_EXAMPLETRAITS_H_
#define EDELSBRUNNER96_EXAMPLETRAITS_H_

#include <vector>
#include <Eigen/Dense>
#include <edelsbrunner96.hpp>

namespace edelsbrunner96 {

/// Example of a traits class suitable for instantiation an edelsbrunner
/// triangulation object
struct ExampleTraits {
  /// dimension of the triangulation
  static const unsigned int NDim = 2;

  /// numeric type for scalars
  typedef double Scalar;

  /// the type used for a NDim point, requires some of the interface of
  /// an Eigen::Matrix. For now that is probably the only option
  typedef Eigen::Matrix<Scalar, NDim, 1> Point;

  /// type which points to a Point structure, this is what is stored inside
  /// simplices. Here we simply use a pointer. Alternatives may include an
  /// integer index into a buffer, or a reference counted pointer.
  /**
   *  Must be a distinct type from SimplexRef to allow for type deduction in
   *  dereferencing. If this is an index into an array, for instance, it
   *  should be a unique type which acts like an index.
   */
  typedef Point* PointRef;

  /// the simplex type, this may be a structure with extra methods and such
  /// but it should provide the same interface as SimplexBase and I strongly
  /// recommend deriving from SimplexBase<Traits> in order to ensure this
  /// requirement
  typedef SimplexBase<ExampleTraits> Simplex;

  /// How we will refer to simplices within the simplex store. For this
  /// example the simplex store is a simple vector so we'll refer to them
  /// by their index
  /**
   *  Must be a distinct type from PointRef to allow for type deduction in
   *  dereferencing. If this is an index into an array, for instance, it
   *  should be a unique type which acts like an index.
   */
  typedef Simplex* SimplexRef;

  /// Storage abstraction for simplices and points.
  /**
   *  Here we actually define a class, but you could just as well typedef a
   *  class that's defined elsewhere.
   *
   *  Must provide
   *    * Promote(SimplexRef)
   *    * Retire(SimplexRef)
   *    * operator[](PointRef)
   *    * operator[](SimplexRef)
   */
  class Storage {
    Point& operator[](PointRef p) {
      return *p;
    }

    PointRef NullPoint() {
      return nullptr;
    }

    Simplex& operator[](SimplexRef s) {
      return *s;
    }

    SimplexRef& NullSimplex() {
      return nullptr;
    }

    SimplexRef Promote() {
      SimplexRef result = nullptr;
      if (!free_.empty()) {
        result = free_.back();
        free_.pop_back();
      } else {
        result = new Simplex();
      }
      return result;
    }

    void Retire(SimplexRef simplex) {
      simplex->version++;
      free_.push_back(simplex);
    }

   private:
    std::vector<Simplex*> free_;
  };
};

}  // namespace edelsbrunner

#endif  // EDELSBRUNNER96_EXAMPLETRAITS_H_
