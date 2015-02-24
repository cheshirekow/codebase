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

#ifndef MPBLOCKS_DUBINS_KD_TREE_DISTANCE_HPP_
#define MPBLOCKS_DUBINS_KD_TREE_DISTANCE_HPP_



namespace mpblocks {
namespace   dubins {
namespace  kd_tree {


template <typename Format>
Format Distance<Format>::operator()( const State_t& q0, const State_t& q1 )
{
    using namespace curves_eigen;
    Path<Format> bestSoln = solve(q0,q1,m_radius);
    return bestSoln.dist(m_radius);
}

template <typename Format>
Format Distance<Format>::operator()( const State_t& q, const Hyper_t& h )
{
    using namespace curves_eigen;
    Path<Format> bestSoln = hyper::solve(q,h,m_radius);
    return bestSoln.dist(m_radius);
}


} // namespace kd_tree
} // namespace dubins
} // namespace mpblocks



#endif // DEFAULTDISTANCE_H_
