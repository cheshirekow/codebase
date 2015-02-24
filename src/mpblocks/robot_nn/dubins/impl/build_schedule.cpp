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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/se3/impl/build_schedule.cpp
 *
 *  @date   Nov 14, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <cmath>
#include "quadtree.h"

namespace mpblocks {
namespace robot_nn {
namespace quadtree {

void build_schedule( std::vector<ScheduleItem>& schedule,
                     float s0, float w, int rate, int depth )
{
    double frac = s0/(2*M_PI*w);
    int    exp;
    std::frexp(frac,&exp);
    int    log2frac = exp-1;

    schedule.clear();
    schedule.reserve(depth);

    for(int i=0; i*rate + log2frac < depth; i++)
    {
        int idx = (i%2 == 0) ? INDEX_R3 : INDEX_SO3;
        schedule.push_back( ScheduleItem(i*rate + log2frac, idx) );
    }
}

}
}
}







