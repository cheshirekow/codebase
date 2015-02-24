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
 *  @file   ListBuilder.h
 *
 *  @date   Feb 17, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_LISTBUILDER_H_
#define MPBLOCKS_KD_TREE_LISTBUILDER_H_

#include <list>
#include <deque>

namespace mpblocks {
namespace  kd_tree {

/// Enumerates an entire subtree, building a list of nodes along with
/// the hyperectangle bounding the subtree at that node
/**
 *  @note lists can be built in breadth first or depth first order
 */
template <class Traits>
class ListBuilder
{
    public:
        typedef typename Traits::Format_t    Format_t;
        typedef typename Traits::Node        Node_t;
        typedef typename Traits::HyperRect   HyperRect_t;

        typedef Eigen::Matrix<Format_t,Traits::NDim,1>  Vector_t;
        typedef Vector_t                                Point_t;

        // these just shorten up some of the templated classes into smaller
        // names
        typedef ListPair<Traits>     Pair_t;
        typedef std::deque<Pair_t*>  Deque_t;
        typedef std::list<Pair_t*>   List_t;

    private:
        Deque_t     m_deque;
        List_t      m_list;
        HyperRect_t m_hyper;

    public:

        /// delete all stored data and get ready for another search
        void reset();

        /// build an enumeration of the tree
        /**
         *  @tparam Inserter_t  type of the insert iterator
         *  @param  root    root of the subtree to build
         *  @param  ins     the inserter where we put nodes we enumerate
         *                  should be an insertion iterator
         */
        template < typename Inserter_t>
        void build( Node_t* root, Inserter_t ins);

        /// enumerate a subtree in breadth-first manner
        void buildBFS( Node_t* root );

        /// enumerate a subtree in depth-first manner
        void buildDFS( Node_t* root );

        /// return the list
        List_t& getList();
};

} // namespace kd_tree
} // namespace mpblocks

#endif /* LISTBUILDER_H_ */
