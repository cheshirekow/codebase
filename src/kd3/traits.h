/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of kd3.
 *
 *  kd3 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  kd3 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with kd3.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef KD3_TRAITS_H_
#define KD3_TRAITS_H_

#include <kd3/euclidean/hyperrect.h>

namespace kd3 {

/// example traits class, can also be used as a default if you're lazy
/// and your problem happens to be 2d
class Traits {
  typedef double Scalar;               ///< number format (i.e. double, float)
  static const unsigned int NDim = 2;  ///< number of dimensions

  /// the hyper rectangle class must be defined in traits before the
  /// Node class since Node uses Traits::HyperRect in it's definition.
  /// For Euclidean kd-tree's however, you may simply typedef this
  typedef euclidean::HyperRect<Traits> HyperRect;

  /// the node class must be defined in traits since it uses the CTRP,
  /// it must derive from kd_tree::Node<Traits> where Traits is the class
  /// containing Node
  class Node : public kd3::Node<Traits> {
   private:
    /// your class implementation may contain extra data
    void* extra_data_;

   public:
    /// your class implementation may have extra methods
    void* GetData();

    /// your class implementation may have extra methods
    void SetData(void* data);
  };
};

}  // namespace kd3

#endif  // KD3_TRAITS_H_
