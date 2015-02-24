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
 *  @file   mpblocks/kd_tree/r2_s1/Distance.h
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_KD_TREE_DISTANCE_H_
#define MPBLOCKS_DUBINS_KD_TREE_DISTANCE_H_



namespace mpblocks {
namespace   dubins {
namespace  kd_tree {

/// provides r2_s1 distance computation
template <typename Format>
class Distance
{
    public:
        typedef typename Traits<Format>::HyperRect  Hyper_t;
        typedef Eigen::Matrix<Format,3,1>           State_t;

    private:
        Format  m_radius;

    public:
        Distance( Format radius=1 ):m_radius(radius){}

        void set_radius( Format radius ){ m_radius = radius; }

        /// return the dubins shortest path distance from point a to point b
        Format    operator()( const State_t& pa, const State_t& pb );

        /// return the dubins shortest path distance from a point to a dubins
        /// state box
        Format    operator()( const State_t& p, const Hyper_t& h );
};



} // namespace kd_tree
} // namespace dubins
} // namespace mpblocks



#endif // DEFAULTDISTANCE_H_
