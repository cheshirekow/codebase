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
 *  @file   mpblocks/kd_tree/r2_s1/KNearest.h
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_KD_TREE_KNEAREST_H_
#define MPBLOCKS_DUBINS_KD_TREE_KNEAREST_H_

#include <vector>
#include <queue>

namespace mpblocks {
namespace   dubins {
namespace  kd_tree {


template <typename Scalar>
class KNearest :
    public mpblocks::kd_tree::NearestSearchIface<Traits<Scalar> >
{
    public:
        typedef typename Traits<Scalar>::Node       Node_t;
        typedef typename Traits<Scalar>::HyperRect  HyperRect_t;

        typedef Distance<Scalar>            Distance_t;
        typedef Key<Scalar>                 Key_t;
        typedef typename Key_t::LessThan    KeyCompare;
        typedef std::vector<Key_t>          KeyVec;

        typedef Eigen::Matrix<Scalar,3,1>                       Point;
        typedef std::priority_queue<Key_t,KeyVec,KeyCompare>    PQueue_t;

        unsigned int    m_nodesEvaluated;
        unsigned int    m_boxesEvaluated;

    protected:
        unsigned int    m_k;
        PQueue_t        m_queue;
        Scalar          m_radius;

    public:
        KNearest( unsigned int k=1, Scalar radius=1 );

        virtual ~KNearest(){};

        // clear the queue
        void reset();

        // clear the queue and change k
        void reset( int k, Scalar radius );

        // return the result
        PQueue_t& result();

        /// calculates Euclidean distance from @p q to @p p, and if its less
        /// than the current best replaces the current best node with @p n
        virtual void evaluate(const Point& q, const Point& p, Node_t* n);

        /// evaluate the Euclidean distance from @p q to it's closest point in
        /// @p r and if that distance is less than the current best distance,
        /// return true
        virtual bool shouldRecurse(const Point& q, const HyperRect_t& r );
};


} // namespace kd_tree
} // namespace dubins
} // namespace mpblocks


#endif // NEARESTNEIGHBOR_H_
