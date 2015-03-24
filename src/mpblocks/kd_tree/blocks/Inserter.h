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
 *  @file   Inserter.h
 *
 *  @date   Sep 28, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */


#ifndef MPBLOCKS_KD_TREE_BLOCKS_INSERTER_H_
#define MPBLOCKS_KD_TREE_BLOCKS_INSERTER_H_

#include <sigc++/sigc++.h>

namespace mpblocks {
namespace  kd_tree {
namespace   blocks {


/// A block which wraps the tree insertion operator.
/**
 *  It provides one slot which
 *  exposes the insert method. When a point is inserted the resulting node
 *  that is created is emitted by the sig_newNode signal
 */
template <class Traits,
            template<class> class Allocator = std::allocator>
class Inserter
{
    public:
        typedef typename Traits::Format_t       Format_t;
        typedef typename Traits::Node           Node_t;

        typedef Allocator<Node_t>                       Allocator_t;
        typedef Eigen::Matrix<Format_t,Traits::NDim,1>  Vector_t;
        typedef Vector_t                                Point_t;

        // these just shorten up some of the templated classes into smaller
        // names
        typedef Tree<Traits>        Tree_t;
        typedef Inserter<Traits>    This_t;
        typedef std::list<Node_t,Allocator_t>   List_t;

    protected:
        Tree_t* m_tree;
        List_t  m_store;

    public:
        /// the signal which is emitted when a new node is created by the tree
        sigc::signal<void,Node_t*>  sig_newNode;

        /// construct an inserter block wrapping the specified tree object
        Inserter( Tree_t* tree = 0 )
            : m_tree(tree)
        {}

        /// change the tree object that this block wraps
        void setTree( Tree_t* tree )
            { m_tree = tree; }

        /// insert the point and emit the new node
        void insert( const Point_t& x )
        {
            m_store.push_back(Node_t());
            Node_t* newNode = &(m_store.back());
            newNode->setPoint(x);
            m_tree->insert(newNode);
            sig_newNode.emit( newNode );
        }

        /// destroy nodes stored
        void reset()
            { m_store.clear(); }

        /// return a slot which calls insert()
        sigc::slot<void,const Point_t&> slot_insert()
            { return sigc::mem_fun(*this,&This_t::insert); }

        /// return a slot which calls reset()
        sigc::slot<void> slot_reset()
            { return sigc::mem_fun(*this,&This_t::reset); }

        /// return const reference to the node set
        List_t& nodes(){ return m_store; }


};




} // namespace blocks
} // namespace kd_tree
} // namespace mpblocks








#endif // INSERTER_H_
