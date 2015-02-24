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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/r2s1/Profile.h
 *
 *  @date   Nov 7, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_ROBOT_NN_EUCLIDEAN_PROFILE_H_
#define MPBLOCKS_ROBOT_NN_EUCLIDEAN_PROFILE_H_


namespace mpblocks {
namespace robot_nn {

struct Profile
{
    int               numTrials;
    int               nodesTouched;
    utility::Timespec runtime;

    Profile():
        numTrials(0),
        nodesTouched(0),
        runtime(0,0)
    {}
};

}
}



#endif // PROFILE_H_
