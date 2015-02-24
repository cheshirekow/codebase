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
 *  @file   /home/josh/Codes/cpp/mpblocks2/gjk/test/demo/MainIface.h
 *
 *  @date   Sep 15, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_MAINIFACE_H_
#define MPBLOCKS_MAINIFACE_H_



namespace mpblocks {
namespace    gjk88 {
namespace     demo {

class MainIface
{
    public:
        virtual ~MainIface(){}
        virtual void run()=0;
};

MainIface* create_main();


} //< namespace demo
} //< namespace gjk88
} //< namespace mpblocks















#endif // MAINIFACE_H_
