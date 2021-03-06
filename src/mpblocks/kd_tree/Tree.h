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
/*
 *  @file   Tree.h
 *
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  basically a c++ rewrite of http://code.google.com/p/kdtree/
 */

#ifndef MPBLOCKS_KD_TREE_TREE_H_
#define MPBLOCKS_KD_TREE_TREE_H_

#include <vector>
#include <memory>

namespace mpblocks {
namespace  kd_tree {

/**
 *  @brief  a simple  KDtree class
 *  @tparam Traits    traits class for kd-tree specifying numer type, dimension
 *                    and the derived class for the node structure
 *  @see    kd_tree:Traits
 */
template <class Traits>
class Tree
{
    public:
        /// number format, i.e. double, float
        typedef typename Traits::Format_t           Format_t;

        /// the node class, should be defined as an inner class of Traits
        typedef typename Traits::Node               Node_t;

        /// the hyper rectangle class shoudl be defined as an inner class of
        /// Traits, or a typedef in Traits
        typedef typename Traits::HyperRect          HyperRect_t;

        /// a vector is the difference of two points
        typedef Eigen::Matrix<Format_t,Traits::NDim,1>  Vector_t;

        /// the storage type for points
        typedef Vector_t                                Point_t;

        // these just shorten up some of the templated classes into smaller
        // names
        typedef Tree<Traits>                        This_t;
        typedef Node<Traits>                        NodeBase_t;
        typedef ListBuilder<Traits >                ListBuilder_t;
        typedef NearestSearchIface<Traits>          NNIface_t;
        typedef RangeSearchIface<Traits>            RangeIface_t;

    protected:
        Node_t*         m_root;     ///< root node of the tree (0 if empty)
        ListBuilder_t   m_lister;   ///< helper for building a list of all nodes
        HyperRect_t     m_rect;     ///< hyper rectangle for searches
        int             m_size;     ///< number of points

        HyperRect_t     m_initRect; ///< initial rectangle, probably 0,0,0...
        HyperRect_t     m_bounds;   ///< bounding rectangle

    public:

        /// constructs a new kd-tree
        Tree( );

        /// destructs the tree and recursively frees all node data.
        /// note that nodes do not own their data if their data are pointers
        ~Tree();

        void set_initRect( const HyperRect_t& h );

        /// insert a node into the kd tree. The node should be newly created
        /// and contain no children
        /**
         *  @param[in]  point   the k-dimensional point to insert into the
         *                      graph
         *  @param[in]  data    the data to store at the newly created node
         *                      (will most-likely contain *point)
         *
         *  @note   The tree does not take ownership of the node poitner
         */
        void insert(Node_t*);

        /// generic NN search, specific search depends on the implementing
        /// class of the NNIface
        void findNearest( const Point_t& q, NNIface_t& search );

        /// generic range search, specific search depends on the implementing
        /// class of the RangeIface
        void findRange( RangeIface_t& search );

        /// create a list of all the nodes in the tree, mostly only used for
        /// debug drawing
        typename ListBuilder_t::List_t& buildList( bool bfs=true );

        /// return the list after buildList has been called, the reference is
        /// the same one returned by buildList but you may want to build the
        /// list and then use it multiple times later
        typename ListBuilder_t::List_t& getList(){ return m_lister.getList(); }

        /// return the root node
        Node_t* getRoot(){ return m_root; }

        /// clearout the data structure, note: does not destroy any object
        /// references
        void clear();

        /// return the number of points in the tree
        int size();
};

} // namespace kd_tree
} // namespace mpblocks


#endif /* CKDTREE_H_ */
