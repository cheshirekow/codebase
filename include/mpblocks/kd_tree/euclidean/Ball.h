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
 *  @file   mpblocks/kd_tree/euclidean/Ball.h
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_BALL_H_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_BALL_H_


namespace  mpblocks {
namespace   kd_tree {
namespace euclidean {


template <class Traits,
            template<class> class Allocator = std::allocator >
class Ball :
    public RangeSearchIface<Traits>
{
    public:
        typedef typename Traits::Format_t   Format_t;
        typedef typename Traits::Node       Node_t;

        typedef Distance<Traits>            Distance_t;
        typedef HyperRect<Traits>           HyperRect_t;
        typedef Key<Traits>                 Key_t;
        typedef typename Key_t::Compare     KeyCompare_t;
        typedef Allocator<Key_t>            Allocator_t;

        typedef Eigen::Matrix<Format_t,Traits::NDim,1>  Point_t;
        typedef std::vector<Node_t*,Allocator_t>        List_t;

    protected:
        Point_t         m_center;
        Format_t        m_radius;
        List_t          m_list;
        Distance_t      m_dist2Fn;

    public:
        Ball( const Point_t& center = Point_t::Zero(),
              Format_t radius = 1 );

        virtual ~Ball(){};

        // clear the queue
        void reset();

        // clear the queue and change k
        void reset( const Point_t& center, Format_t radius );

        // return the result
        const List_t& result();

        /// calculates Euclidean distance from @p q to @p p, and if its less
        /// than the current best replaces the current best node with @p n
        virtual void evaluate(const Point_t& p, Node_t* n);

        /// evaluate the Euclidean distance from @p q to it's closest point in
        /// @p r and if that distance is less than the current best distance,
        /// return true
        virtual bool shouldRecurse(const HyperRect_t& r );
};






} // namespace euclidean
} // namespace kd_tree
} // namespace mpblocks








#endif // NEARESTNEIGHBOR_H_
