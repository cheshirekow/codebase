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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/so3.h
 *
 *  @date   Nov 12, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_ROBOT_NN_SE3_H_
#define MPBLOCKS_ROBOT_NN_SE3_H_

#include <cmath>
#include "so3.h"

namespace mpblocks {
namespace robot_nn {
namespace      se3 {


template <typename Scalar>
Scalar se3_pseudo_distance(
        Scalar weight,
        const Eigen::Matrix<Scalar,7,1>& q_se3_0,
        const Eigen::Matrix<Scalar,7,1>& q_se3_1 )
{
    Eigen::Matrix<Scalar,3,1> x0 = q_se3_0.block(0,0,3,1);
    Eigen::Matrix<Scalar,3,1> x1 = q_se3_1.block(0,0,3,1);
    Eigen::Matrix<Scalar,4,1> q0 = q_se3_0.block(3,0,4,1);
    Eigen::Matrix<Scalar,4,1> q1 = q_se3_1.block(3,0,4,1);

    return (x1-x0).normSquared() + w*so3::so3_pseudo_distance(q0,q1);
}

template <typename Scalar>
Scalar se3_distance(
        Scalar weight,
        const Eigen::Matrix<Scalar,7,1>& q_se3_0,
        const Eigen::Matrix<Scalar,7,1>& q_se3_1 )
{
    Eigen::Matrix<Scalar,3,1> x0 = q_se3_0.block(0,0,3,1);
    Eigen::Matrix<Scalar,3,1> x1 = q_se3_1.block(0,0,3,1);
    Eigen::Matrix<Scalar,4,1> q0 = q_se3_0.block(3,0,4,1);
    Eigen::Matrix<Scalar,4,1> q1 = q_se3_1.block(3,0,4,1);

    return (x1-x0).norm() + w*so3::so3_distance(q0,q1);
}


template <typename Scalar, int NDim>
struct HyperRect
{
    Eigen::Matrix<Scalar,NDim,1> data[2];

    Eigen::Matrix<Scalar,NDim,1>& operator[](int i)
    {
        return data[i];
    }

    const Eigen::Matrix<Scalar,NDim,1>& operator[](int i) const
    {
        return data[i];
    }
};


template <typename Scalar, class Hyper>
void se3_pseudo_distance(
        Scalar weight,
        const Eigen::Matrix<Scalar,7,1>& q,
        const Hyper& H,
        Scalar& dist,
        bool& feasible )
{
    // split out q-rect and q-query
    Eigen::Matrix<Scalar,4,1> q_q = q.block(3,0,4,1);
    HyperRect<Scalar,4> q_rect;
    for(int i=0; i < 2; i++)
    {
        for(int j=3; j < 7; j++)
            q_rect[i][j-3] = H[i][j];
    }
    so3::so3_pseudo_distance(q_q,q_rect,dist,feasible);
    dist *= w;

    Scalar dist_x=0;
    for(int i=0; i < 3; i++)
    {
        Scalar dist_i = 0;
        if( q[i] < H[0][i] )
            dist_i = H[0][i] - q[i];
        else if( q[i] > H[1][i] )
            dist_i = q[i] - H[1][i];
        dist_x += dist_i*dist_i;
    }

    dist += dist_x;
}


template <typename Scalar, class Hyper>
void se3_distance(
        Scalar weight,
        const Eigen::Matrix<Scalar,7,1>& q,
        const Hyper& H,
        Scalar& dist,
        bool& feasible )
{
    // split out q-rect and q-query
    Eigen::Matrix<Scalar,4,1> q_q = q.block(3,0,4,1);
    HyperRect<Scalar,4> q_rect;
    for(int i=0; i < 2; i++)
    {
        for(int j=3; j < 7; j++)
            q_rect[i][j-3] = H[i][j];
    }
    so3::so3_distance(q_q,q_rect,dist,feasible);
    dist *= weight;

    Scalar dist_x=0;
    for(int i=0; i < 3; i++)
    {
        Scalar dist_i = 0;
        if( q[i] < H[0][i] )
            dist_i = H[0][i] - q[i];
        else if( q[i] > H[1][i] )
            dist_i = q[i] - H[1][i];
        dist_x += dist_i*dist_i;
    }

    dist += std::sqrt(dist_x);
}

template <typename Scalar, class Hyper>
void se3_distance_debug(
        const Eigen::Matrix<Scalar,7,1>& q,
        const Hyper& H,
        Scalar& dist_x,
        Scalar& dist_q,
        bool& feasible )
{
    // split out q-rect and q-query
    Eigen::Matrix<Scalar,4,1> q_q = q.block(3,0,4,1);
    HyperRect<Scalar,4> q_rect;
    for(int i=0; i < 2; i++)
    {
        for(int j=3; j < 7; j++)
            q_rect[i][j-3] = H[i][j];
    }
    so3::so3_distance(q_q,q_rect,dist_q,feasible);

    dist_x=0;
    for(int i=0; i < 3; i++)
    {
        Scalar dist_i = 0;
        if( q[i] < H[0][i] )
            dist_i = H[0][i] - q[i];
        else if( q[i] > H[1][i] )
            dist_i = q[i] - H[1][i];
        dist_x += dist_i*dist_i;
    }

    dist_x = std::sqrt(dist_x);
}



} //< namespace so3
} //< namespace robot_nn
} //< namespace mpblocks















#endif // SO3_H_
