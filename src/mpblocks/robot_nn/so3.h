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

#ifndef MPBLOCKS_ROBOT_NN_SO3_H_
#define MPBLOCKS_ROBOT_NN_SO3_H_

#include <cmath>
#include <Eigen/Dense>

namespace mpblocks {
namespace robot_nn {
namespace      so3 {

enum Constraint
{
    OFF = 0,
    MIN = 1,
    MAX = 2
};

struct ConstraintSpec
{
    char m_data[4];

    ConstraintSpec( char c0,
                    char c1,
                    char c2,
                    char c3 )
    {
        m_data[0] = c0;
        m_data[1] = c1;
        m_data[2] = c2;
        m_data[3] = c3;
    }

    ConstraintSpec( int idx )
    {
        m_data[3] = idx % 3; idx/=3;
        m_data[2] = idx % 3; idx/=3;
        m_data[1] = idx % 3; idx/=3;
        m_data[0] = idx;
    }

    int toIdx()
    {
        return
            m_data[0] + 3*(
            m_data[1] + 3*(
            m_data[2] + 3*(
            m_data[3] )));
    }

    char operator[]( int i ) const
    {
        return m_data[i];
    }

    bool operator==( const ConstraintSpec& other ) const
    {
        for(int i=0; i < 4; i++)
            if( m_data[i] != other.m_data[i] )
                return false;
        return true;
    }
};


template <typename Scalar>
Scalar so3_pseudo_distance(
        const Eigen::Matrix<Scalar,4,1>& q0,
        const Eigen::Matrix<Scalar,4,1>& q1 )
{
    Scalar dq = q0.dot(q1);
    dq = std::max(Scalar(-1),std::min(dq,Scalar(1)));
    return 1-dq;
}

template <typename Scalar>
Scalar so3_distance(
        const Eigen::Matrix<Scalar,4,1>& q0,
        const Eigen::Matrix<Scalar,4,1>& q1 )
{
    Scalar dot = q0.dot(q1);
    Scalar arg = 2*dot*dot - 1;
    return std::acos(arg);
}



template <int Sign, typename Scalar, class HyperRect>
void so3_pseudo_distance(
        const ConstraintSpec& spec,
        const Eigen::Matrix<Scalar,4,1>& q,
        const HyperRect& H,
        Scalar& dist,
        bool&   feasible )
{
    dist = Scalar(10.0);

    if( spec == ConstraintSpec(OFF,OFF,OFF,OFF) )
    {
        feasible = true;
        for(int i=0; i < 4; i++)
            if( q[i] < H[0][i] || q[i] > H[1][i] )
                feasible = false;
        if(feasible)
            dist = 0;
        return;
    }

    Scalar den = 1;
    Scalar num = 0;

    for(int i=0; i < 4; i++)
    {
        switch(spec[i])
        {
            case MIN:
                den -= H[0][i]*H[0][i];
                break;
            case MAX:
                den -= H[1][i]*H[1][i];
                break;
            default:
                num += q[i]*q[i];
                break;
        }
    }

    if( den < 1e-8 )
    {
        feasible = false;
        return;
    }

    Scalar lambda2  = num / (4*den);
    Scalar lambda   = std::sqrt(lambda2);
    feasible        = true;

    Eigen::Matrix<Scalar,4,1> x;
    for(int i=0; i < 4; i++)
    {
        switch(spec[i])
        {
            case MIN:
                x[i] = H[0][i];
                break;
            case MAX:
                x[i] = H[1][i];
                break;
            default:
                x[i] = -q[i] / (2*Sign*lambda);
                if( x[i] > H[1][i] || x[i] < H[0][i] )
                    feasible =false;
                break;
        }
    }
    dist = so3_pseudo_distance(q,x);
}



template <typename Scalar, class HyperRect>
void so3_pseudo_distance(
        const Eigen::Matrix<Scalar,4,1>& q,
        const HyperRect& H,
        Scalar& dist,
        bool&   feasible )
{
    dist     = 1e9;
    feasible = false;
    for(int i=0; i < 81; i++)
    {
        float dist_i     = 1e9;
        bool  feasible_i = false;
        so3_pseudo_distance<-1>(
                ConstraintSpec(i),q,H,dist_i,feasible_i );
        if( feasible_i && dist_i < dist )
        {
            dist = dist_i;
            feasible = true;
        }

        so3_pseudo_distance<1>(
                ConstraintSpec(i),q,H,dist_i,feasible_i );
        if( feasible_i && dist_i < dist )
        {
            dist = dist_i;
            feasible = true;
        }
    }
}


template <int Sign, typename Scalar, class HyperRect>
void so3_distance(
        const ConstraintSpec& spec,
        const Eigen::Matrix<Scalar,4,1>& q,
        const HyperRect& H,
        Scalar& dist,
        bool&   feasible )
{
    dist = Scalar(10.0);

    if( spec == ConstraintSpec(OFF,OFF,OFF,OFF) )
    {
        feasible = true;
        for(int i=0; i < 4; i++)
            if( q[i] < H[0][i] || q[i] > H[1][i] )
                feasible = false;
        if(feasible)
            dist = 0;
        return;
    }

    Scalar den = 1;
    Scalar num = 0;

    for(int i=0; i < 4; i++)
    {
        switch(spec[i])
        {
            case MIN:
                den -= H[0][i]*H[0][i];
                break;
            case MAX:
                den -= H[1][i]*H[1][i];
                break;
            default:
                num += q[i]*q[i];
                break;
        }
    }

    if( den < 1e-8 )
    {
        feasible = false;
        return;
    }

    Scalar lambda2  = num / (4*den);
    Scalar lambda   = std::sqrt(lambda2);
    feasible        = true;

    Eigen::Matrix<Scalar,4,1> x;
    for(int i=0; i < 4; i++)
    {
        switch(spec[i])
        {
            case MIN:
                x[i] = H[0][i];
                break;
            case MAX:
                x[i] = H[1][i];
                break;
            default:
                x[i] = -q[i] / (2*Sign*lambda);
                if( x[i] > H[1][i] || x[i] < H[0][i] )
                    feasible =false;
                break;
        }
    }
    dist = so3_distance(q,x);
}

template <typename Scalar, class HyperRect>
void so3_distance(
        const Eigen::Matrix<Scalar,4,1>& q,
        const HyperRect& H,
        Scalar& dist,
        bool&   feasible )
{
    dist     = 1e9;
    feasible = false;
    for(int i=0; i < 81; i++)
    {
        float dist_i     = 1e9;
        bool  feasible_i = false;
        so3_distance<-1>(
                ConstraintSpec(i),q,H,dist_i,feasible_i );
        if( feasible_i && dist_i < dist )
        {
            dist = dist_i;
            feasible = true;
        }

        so3_distance<1>(
                ConstraintSpec(i),q,H,dist_i,feasible_i );
        if( feasible_i && dist_i < dist )
        {
            dist = dist_i;
            feasible = true;
        }
    }
}


} //< namespace so3
} //< namespace robot_nn
} //< namespace mpblocks















#endif // SO3_H_
