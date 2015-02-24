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
 *  @file   mpblocks/kd_tree/r2_s1/KNearest.hpp
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_KD_TREE_KNEAREST_HPP_
#define MPBLOCKS_DUBINS_KD_TREE_KNEAREST_HPP_


namespace  mpblocks {
namespace    dubins {
namespace   kd_tree {


template <typename Scalar>
KNearest<Scalar>::KNearest(unsigned int k, Scalar radius)
{
    m_k = k;
    m_radius = radius;

    reset();
}


template <typename Scalar>
void KNearest<Scalar>::reset(  )
{
    while( m_queue.size() > 0 )
        m_queue.pop();
    m_nodesEvaluated = 0;
    m_boxesEvaluated = 0;
}


template <typename Scalar>
void KNearest<Scalar>::reset( int k, Scalar radius )
{
    reset();
    m_k = k;
    m_radius = radius;
}



template <typename Scalar>
typename KNearest<Scalar>::PQueue_t& KNearest<Scalar>::result()
{
    return m_queue;
}




template <typename Scalar>
void KNearest<Scalar>::evaluate(
        const Point& q0, const Point& q1, Node_t* n)
{
    m_nodesEvaluated++;

    using namespace curves_eigen;
    Path<Scalar> bestSoln = solve(q0,q1,m_radius);

    Key_t    key;
    key.d2 = bestSoln.dist(m_radius);
    key.n  = n;
    key.id = bestSoln.id;

    m_queue.push(key);

    if( m_queue.size() > m_k )
        m_queue.pop();
}




template <typename Scalar>
bool KNearest<Scalar>::shouldRecurse( const Point& q, const HyperRect_t& h )
{
    m_boxesEvaluated++;
//    return true;
    using namespace curves_eigen;
    Path<Scalar> bestSoln = hyper::solve(q,h,m_radius);
    Scalar d2 = bestSoln.dist(m_radius);

    return ( m_queue.size() < m_k || d2 < m_queue.top().d2 );
}







} // namespace kd_tree
} // namespace dubins
} // namespace mpblocks



#endif // NEARESTNEIGHBOR_H_
