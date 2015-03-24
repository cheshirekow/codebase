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
 *  @file   mpblocks/simplex_tree/Simplex.h
 *
 *  @date   Dec 2, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_SIMPLEX_TREE_SIMPLEX_H_
#define MPBLOCKS_SIMPLEX_TREE_SIMPLEX_H_

#include <Eigen/Dense>
#include <vector>
#include <list>

namespace mpblocks     {
namespace simplex_tree {

/// base class for nodes, implements storage and interface for the simplex tree
/**
 *  A node is something which is inserted into the tree. It must provide a
 *  notion of a point in @f \mathbb{R}^n @f.
 *
 *  The proper usage is to define a member class of Traits which is the
 *  actual node class, and which derives from this class. By CTRP it will
 *  contain storage and the necessary interface for the simplex tree, but the
 *  derived class is always returned through API calls
 */
template <class Traits>
class Simplex
{
    public:
        typedef Simplex<Traits>                 Simplex_t;
        typedef typename Traits::Format         Format_t;
        typedef typename Traits::Node           Node_t;

        typedef unsigned int                                        Index_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim,1>              Vector_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim,1>              Point_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim+1,Traits::NDim> Normals_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim+1,1>            Offsets_t;

        typedef std::vector<Simplex_t*>     SimplexList_t;
        typedef std::vector<Node_t*>        NodeList_t;

    protected:
        Index_t     m_idx;  /// index of the vertex of the parent which is
                            /// replaced in creating this node
        Point_t     m_v;    /// vertex of the parent which is replaced
                            /// in creating this node

        Normals_t   m_n;    /// normals of the faces
        Offsets_t   m_d;    /// offsets of the faces

        SimplexList_t m_children;  ///< list of children if it's an interior
                                   ///< simplex
        SimplexList_t m_neighbors; ///< list of neighbors

        NodeList_t  m_points;   /// list of points in this node, if it's a
                                /// leaf

        /// split the node according to the split policy converting it from a
        /// leaf node to an interior node
        void split();

    public:
        bool isInterior(){ return m_children.size() == 0; }
        bool isLeaf()    { return !isInterior(); }

        /// initialize the simplex as leaf, reserving storage for all the
        /// child node pointers
        void initAsLeaf(Simplex_t* parent, Index_t i, Point_t& v);

        /// return true if the simplex contains the point
        bool contains( Node_t* );

        /// insert the point into the simplex, recursively inserting into
        /// children or splitting as necessary
        void insert( Node_t* );

        /// set the i'th face of this simplex with the specified normal and
        /// distance
        void setFace(Index_t i, const Vector_t& n, Format_t d );




};






}   // namespace simplex_tree
}   // namespace mpblocks












#endif // SIMPLEX_H_
