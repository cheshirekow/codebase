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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/r2s1/Implementation.h
 *
 *  @date   Nov 7, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_ROBOT_NN_R2S1_IMPLEMENTATION_H_
#define MPBLOCKS_ROBOT_NN_R2S1_IMPLEMENTATION_H_

#include <map>
#include <mpblocks/utility/Timespec.h>

#include "config.h"
#include "Profile.h"

namespace mpblocks {
namespace robot_nn {
namespace   dubins {




/// interface for different implementations
struct Implementation
{
    typedef Eigen::Matrix<float,3,1> Point;

    int size;
    int touched;
    float m_r;
    std::map<int,Profile> profiles;

    Implementation():
        size(0),
        touched(0)
    {
        m_r = 1.0;
    }

    virtual ~Implementation(){}

    virtual void allocate( int maxSize )=0;
    virtual void insert_derived( int i, const Point& x )=0;
    virtual void findNearest_derived( const Point& q )=0;
    virtual void get_result( std::vector<int>& out )=0;

    void insert( int i, const Point& x )
    {
        insert_derived(i,x);
        size++;
    }

    void findNearest( const Point& q )
    {
        utility::Timespec sw_start, sw_stop;
        clock_gettime(CLOCK_MONOTONIC,&sw_start);
        findNearest_derived(q);
        clock_gettime(CLOCK_MONOTONIC,&sw_stop);
        Profile& profile = profiles[size];
        profile.runtime      += (sw_stop-sw_start);
        profile.nodesTouched += touched;
        profile.numTrials    ++;
    }


};





}
}
}





#endif // IMPLEMENTATION_H_
