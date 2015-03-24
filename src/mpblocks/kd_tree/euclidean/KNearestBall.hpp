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
 *  @file   mpblocks/kd_tree/euclidean/KNearestBallBall.hpp
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_KNEARESTBALL_HPP_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_KNEARESTBALL_HPP_


namespace  mpblocks {
namespace   kd_tree {
namespace euclidean {


template <class Traits,template<class> class Allocator>
KNearestBall<Traits,Allocator>::KNearestBall(Format_t radius, unsigned int k)
{
    reset(radius,k);
}


template <class Traits,template<class> class Allocator>
void KNearestBall<Traits,Allocator>::reset(  )
{
    m_queue.clear();
}


template <class Traits,template<class> class Allocator>
void KNearestBall<Traits,Allocator>::reset( Format_t radius, int k )
{
    m_radius = radius;
    m_k      = k;
    m_queue.clear();
}



template <class Traits,template<class> class Allocator>
const typename KNearestBall<Traits,Allocator>::PQueue_t&
    KNearestBall<Traits,Allocator>::result()
{
    return m_queue;
}




template <class Traits,template<class> class Allocator>
void KNearestBall<Traits,Allocator>::evaluate(
        const Point_t& q, const Point_t& p, Node_t* n)
{
    Format_t r2 = m_radius*m_radius;
    Format_t d2 = m_dist2Fn(q,p);

    if( d2 < r2 )
    {
        Key_t    key;
        key.d2 = d2;
        key.n  = n;

        m_queue.insert(key);

        if( m_queue.size() > m_k )
            m_queue.erase( --m_queue.end() );
    }


}




template <class Traits,template<class> class Allocator>
bool KNearestBall<Traits,Allocator>::shouldRecurse(
        const Point_t& q, const HyperRect_t& r )
{
    // check if we have to recurse into the farther subtree by
    // finding out if the nearest point in the hyperrectangle is
    // closer to the query point then the current k'th best guess
    // or if there are less then k nearest in the queue
    Format_t dist2      = m_dist2(q,r);
    bool isInRange      = dist2 < m_radius*m_radius;
    bool isCloser       = dist2 < m_queue.rbegin()->d2;
    bool isAvailability = m_queue.size() < m_k;

    return (  isInRange && ( isCloser || isAvailability) );
}






} // namespace euclidean
} // namespace kd_tree
} // namespace mpblocks








#endif // NEARESTNEIGHBOR_H_
