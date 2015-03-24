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
 *  @file   mpblocks/kd_tree/euclidean/Distance.hpp
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_DISTANCE_HPP_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_DISTANCE_HPP_



namespace  mpblocks {
namespace   kd_tree {
namespace euclidean {

template <class Traits>
typename Traits::Format_t
Distance<Traits>::operator()( const Point_t& pa, const Point_t& pb )
{
    return (pa-pb).squaredNorm();
}

template <class Traits>
typename Traits::Format_t
Distance<Traits>::operator()( const Point_t& p, const Hyper_t& h )
{
    Format_t dist2 = 0;
    Format_t dist2i = 0;

    for (unsigned int i=0; i < p.rows(); i++)
    {
        if (p[i] < h.minExt[i])
            dist2i = h.minExt[i] - p[i];
        else if(p[i] > h.maxExt[i])
            dist2i = h.maxExt[i] - p[i];
        else
            dist2i = 0;

        dist2i *= dist2i;
        dist2 += dist2i;
    }

    return dist2;
}



} // namespace euclidean
} // namespace kd_tree
} // namespace mpblocks



#endif // DEFAULTDISTANCE_H_
