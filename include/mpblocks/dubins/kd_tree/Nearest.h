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
 *  @file   mpblocks/kd_tree/r2_s1/Nearest.h
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_KD_TREE_NEAREST_H_
#define MPBLOCKS_DUBINS_KD_TREE_NEAREST_H_


namespace mpblocks {
namespace   dubins {
namespace  kd_tree {


template <typename Scalar>
class Nearest :
    public mpblocks::kd_tree::NearestSearchIface< Traits<Scalar> >
{
    public:
        typedef typename Traits<Scalar>::Node       Node_t;
        typedef typename Traits<Scalar>::HyperRect  HyperRect_t;

        typedef Distance<Scalar>            Distance_t;
        typedef Eigen::Matrix<Scalar,3,1>   Point;

    private:
        Distance_t  m_dist2Fn;
        Scalar      m_d2Best;
        Node_t*     m_nBest;

    public:
        virtual ~Nearest(){};

        // reset d2Best to be infinity
        void reset(  );

        // return the result
        Node_t* result();

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
