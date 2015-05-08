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
 *  \file   mpblocks/kd_tree/Traits.h
 *
 *  \date   Nov 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief
 */

#ifndef MPBLOCKS_KD_TREE_TRAITS_H_
#define MPBLOCKS_KD_TREE_TRAITS_H_

namespace mpblocks {
namespace kd_tree {

/// example traits class, can also be used as a default if you're lazy
/// and your problem happens to be 2d
class Traits {
  typedef double Format_t;             ///< number format (i.e. double, float)
  static const unsigned int NDim = 2;  ///< number of dimensions

  /// the hyper rectangle class must be defined in traits before the
  /// Node class since Node uses Traits::HyperRect in it's definition.
  /// For Euclidean kd-tree's however, you may simply typedef this
  typedef euclidean::HyperRect<Traits> HyperRect;

  /// the node class must be defined in traits since it uses the CTRP,
  /// it must derive from kd_tree::Node<Traits> where Traits is the class
  /// containing Node
  class Node : public kd_tree::Node<Traits> {
   private:
    /// your class implementation may contain extra data
    void* m_extraData;

   public:
    /// your class implementation may have extra methods
    void* getData();

    /// your class implementation may have extra methods
    void setData(void* data);
  };
};

}  // namespace kd_tree
}  // namespace mpblocks

#endif  // TRAITS_H_
