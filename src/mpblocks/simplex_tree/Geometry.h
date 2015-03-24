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
 *  @file   mpblocks/simplex_tree/Geometry.h
 *
 *  @date   Dec 2, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_GEOMETRY_TREE_GEOMETRY_H_
#define MPBLOCKS_GEOMETRY_TREE_GEOMETRY_H_

#include <Eigen/Dense>
#include <vector>
#include <list>
#include <bitset>

namespace mpblocks     {
namespace simplex_tree {

/// provides vertex storage for recursively building the "active" simplex
/// during insertion
template <class Traits>
class Geometry
{
    public:
        typedef Simplex<Traits>                 Simplex_t;
        typedef typename Traits::Format         Format_t;
        typedef typename Traits::Node           Node_t;

        typedef unsigned int                                        Index_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim,1>              Vector_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim,1>              Point_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim,Traits::NDim+1> Vertices_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim+1,Traits::NDim> Normals_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim+1,1>            Offsets_t;
        typedef std::bitset<Traits::NDim+1>                         Bitset_t;

    protected:
        static const Format_t  sm_n;

        Bitset_t    m_isSet;    ///< whether or not the i'th vertex is set
        Vertices_t  m_v;        ///< storage for vertices

    public:
        /// make infinite
        void reset();

        /// refine by replacing the i'th vertex with the specified point
        void refine( Index_t i, const Point_t& v );

        /// build a set of faces based on the current simplex, appropriately
        /// handling vertices at infinity
        void initFaces(  );


};






}   // namespace simplex_tree
}   // namespace mpblocks












#endif // GEOMETRY_H_
