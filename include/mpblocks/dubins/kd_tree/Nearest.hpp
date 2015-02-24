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
 *  @file   mpblocks/kd_tree/r2_s1/Nearest.hpp
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_KD_TREE_NEAREST_HPP_
#define MPBLOCKS_DUBINS_KD_TREE_NEAREST_HPP_


namespace  mpblocks {
namespace    dubins {
namespace   kd_tree {




template <typename Scalar>
void Nearest<Scalar>::reset(  )
{
    m_d2Best = 0;
    m_nBest  = 0;
}



template <typename Scalar>
typename Traits<Scalar>::Node* Nearest<Scalar>::result()
{
    return m_nBest;
}




template <typename Scalar>
void Nearest<Scalar>::evaluate( const Point& q, const Point& p, Node_t* n)
{
    Scalar d2 = m_dist2Fn(q,p);
    if( d2 < m_d2Best || !m_nBest )
    {
        m_d2Best = d2;
        m_nBest  = n;
    }
}




template <typename Scalar>
bool Nearest<Scalar>::shouldRecurse( const Point& q, const HyperRect_t& r )
{
    Scalar d2  = m_dist2Fn(q,r);
    return (d2 < m_d2Best);
}






} // namespace kd_tree
} // namespace dubins
} // namespace mpblocks








#endif // NEARESTNEIGHBOR_H_
