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
 *  @file   ListBuilder.cpp
 *
 *  @date   Feb 17, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_LISTBUILDER_HPP_
#define MPBLOCKS_KD_TREE_LISTBUILDER_HPP_


namespace mpblocks {
namespace  kd_tree {




template <class Traits>
void ListBuilder<Traits>::reset()
{
    for( typename Deque_t::iterator ipPair = m_deque.begin();
            ipPair != m_deque.end(); ipPair++ )
        delete *ipPair;

    for( typename List_t::iterator ipPair = m_list.begin();
            ipPair != m_list.end(); ipPair++ )
        delete *ipPair;

    m_list.clear();
    m_deque.clear();
    m_hyper.makeInfinite();
}





template <class Traits>
template < typename Inserter_t>
void ListBuilder<Traits>::build( Node_t* root, Inserter_t ins)
{
    Pair_t* rootPair = new Pair_t();
    rootPair->node = root;
    m_hyper.copyTo(rootPair->container);
    m_deque.push_back(rootPair);

    while(m_deque.size() > 0)
    {
        Pair_t* pair = m_deque.front();
        m_deque.pop_front();
        m_list.push_back(pair);
        pair->node->enumerate(pair->container, ins );
    }
}




template <class Traits>
void ListBuilder<Traits>::buildBFS( Node_t* root )
{
    build( root, std::back_inserter(m_deque) );
}





template <class Traits>
void ListBuilder<Traits>::buildDFS( Node_t* root )
{
    build( root, std::front_inserter(m_deque) );
}



template <class Traits>
std::list<ListPair<Traits>*>& ListBuilder<Traits>::getList()
{
    return m_list;
}






} // namespace kd_tree
} // namespace mpblocks


#endif
