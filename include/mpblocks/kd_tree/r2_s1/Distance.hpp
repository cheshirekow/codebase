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
 *  @file   mpblocks/kd_tree/r2_s1/Distance.hpp
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_KD_TREE_R2_S1_DISTANCE_HPP_
#define MPBLOCKS_KD_TREE_R2_S1_DISTANCE_HPP_



namespace mpblocks {
namespace  kd_tree {
namespace    r2_s1 {

template <class Traits>
Distance<Traits>::Distance():
    m_min(0),
    m_max(1),
    m_weight(1.0)
{}

template <class Traits>
Distance<Traits>::Distance(Format_t min, Format_t max, Format_t weight):
    m_min(min),
    m_max(max),
    m_weight(weight)
{}

template <class Traits>
void Distance<Traits>::configure(Format_t min, Format_t max, Format_t weight)
{
    m_min = min;
    m_max = max;
    m_weight = weight;
}




template <class Traits>
typename Traits::Format_t
Distance<Traits>::operator()( const Point_t& pa, const Point_t& pb )
{
    // create a copy of one of the points where the s1 coordinate has been
    // wrapped around. If s_1 is mapped to [a,b] and pb[2] is c then
    // [a---c----b] -> [a-------b----(b+c-a)]
    Point_t pb_wrap = pb;
    pb_wrap[2] = m_max + (pb[2] - m_min);

    Point_t diff_a = (pb-pa);
    Point_t diff_b = (pb_wrap-pa);

    Format_t d2a = diff_a[0]*diff_a[0]
                    + diff_a[1]*diff_a[1]
                    + m_weight * diff_a[2] * diff_a[2];

    Format_t d2b = diff_b[0]*diff_b[0]
                    + diff_b[1]*diff_b[1]
                    + m_weight * diff_b[2] * diff_b[2];

    return std::min( d2a,d2b );
}

template <class Traits>
typename Traits::Format_t
Distance<Traits>::operator()( const Point_t& p, const Hyper_t& h )
{
    Format_t dist2  = 0;
    Format_t dist2i, dist2w;

    for (unsigned int i=0; i < 2; i++)
    {
        if (p[i] < h.minExt[i])
            dist2i = h.minExt[i] - p[i];
        else if(p[i] > h.maxExt[i])
            dist2i = h.maxExt[i] - p[i];
        else
            continue;

        dist2 += dist2i * dist2i;
    }

    // distance of p[2] to an interval [xa,xb] as in
    // [x0-----p2----xa----xb----x1]
    if( p[2] < h.minExt[2] )
    {
        dist2i = h.minExt[2] - p[2];
        dist2w = (m_max + p[2] - m_min ) - h.maxExt[2];

        if( dist2w < dist2i )
            dist2i = dist2w;

        dist2 += m_weight * dist2i * dist2i;
    }
    else if( p[2] > h.maxExt[2] )
    {
        dist2i = p[2] - h.maxExt[2];
        dist2w = (m_max + h.minExt[2] - m_min) - p[2];

        if( dist2w < dist2i )
            dist2i = dist2w;

        dist2 += m_weight * dist2i * dist2i;
    }

    return dist2;
}



} // namespace r2_s1
} // namespace kd_tree
} // namespace mpblocks



#endif // DEFAULTDISTANCE_H_
