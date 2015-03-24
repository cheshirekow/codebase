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
 *  @file   mpblocks/simplex_tree/Simplex.hpp
 *
 *  @date   Dec 2, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_SIMPLEX_TREE_SIMPLEX_HPP_
#define MPBLOCKS_SIMPLEX_TREE_SIMPLEX_HPP_


namespace mpblocks     {
namespace simplex_tree {

template <class Traits>
void Simplex<Traits>::split()
{
    // reserve storage for all the children and create the child simplex
    // structures
    Index_t n = Traits::NDim+1;
    m_children.reserve(n);
    for(int i=0; i < n; i++)
        m_children.push_back( new Simplex_t() );

    // for each of the N+1 points find the face midway between all other points
    for(Index_t i=0; i < n; i++)
    {
        const Point_t& xi = m_points[i]->getPoint();

        // copy the i'th face from the parent
        //

        // all other faces are found by the plane separating the i'th
        // from one of the other points
        for(Index_t j=i; j < n; j++)
        {
            const Point_t& xj = m_points[i]->getPoints();
            Vector_t v   = xj - xi;
            Vector_t nj  = v.normalized();
            Format_t dj  = nj.dot(xi) + 0.5*v.norm();
        }
    }

    // now clear our point list and force the vector to release it's memory
    m_points.clear();
    NodeList_t tmp; tmp.swap(m_points);
}


template <class Traits>
void Simplex<Traits>::initAsLeaf(Simplex_t* parent, Index_t i, Point_t& v)
{

}


template <class Traits>
void Simplex<Traits>::insert( Node_t* x )
{
    if(isLeaf())
    {
        m_points.push_back(x);
        if(m_points.size() > Traits::NDim+1 )
            split();
    }
    else
    {
        const int n = m_children.size();
        for(int i=0; i < n; i++)
        {
            if( m_children[i]->contains(x) )
            {
                m_children[i]->insert(x);
                break;
            }
        }
    }
}





}   // namespace simplex_tree
}   // namespace mpblocks



#endif // SIMPLEX_HPP_
