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
 *  @file   mpblocks/kd_tree/euclidean/Ball.hpp
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_BALL_HPP_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_BALL_HPP_


namespace  mpblocks {
namespace   kd_tree {
namespace euclidean {


template <class Traits,template<class> class Allocator>
Ball<Traits,Allocator>::Ball(const Point_t& center, Format_t radius)
{
    reset(center,radius);
}


template <class Traits,template<class> class Allocator>
void Ball<Traits,Allocator>::reset(  )
{
    m_list.clear();
}


template <class Traits,template<class> class Allocator>
void Ball<Traits,Allocator>::reset( const Point_t& center, Format_t radius )
{
    m_center = center;
    m_radius = radius;
    m_list.clear();
}



template <class Traits,template<class> class Allocator>
const typename Ball<Traits,Allocator>::List_t&
    Ball<Traits,Allocator>::result()
{
    return m_list;
}




template <class Traits,template<class> class Allocator>
void Ball<Traits,Allocator>::evaluate(const Point_t& p, Node_t* n)
{
    Format_t r2 = m_radius*m_radius;
    Format_t d2 = m_dist2Fn(p,m_center);

    if( d2 < r2 )
        m_list.push_back(n);
}




template <class Traits,template<class> class Allocator>
bool Ball<Traits,Allocator>::shouldRecurse(const HyperRect_t& r )
{
    // check if we have to recurse into the farther subtree by
    // finding out if the nearest point in the hyperrectangle is
    // inside the ball
    Format_t dist2      = m_dist2Fn(m_center,r);
    return (dist2 < m_radius*m_radius);
}






} // namespace euclidean
} // namespace kd_tree
} // namespace mpblocks








#endif // NEARESTNEIGHBOR_H_
