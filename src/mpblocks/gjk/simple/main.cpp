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
 *  @file   /home/josh/Codes/cpp/mpblocks2/gjk/test/simple/main.cpp
 *
 *  @date   Sep 14, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <iostream>
#include <mpblocks/gjk88.h>
#include <Eigen/Dense>

using namespace mpblocks;

typedef Eigen::Vector3d Point;

struct Policy
{
    Point cross( const Point& a, const Point& b )
    {
        return a.cross(b);
    }

    double dot( const Point& a, const Point& b )
    {
        return a.dot(b);
    }

    bool threshold( const Point& a, const Point& b )
    {
        return true;
    }

    bool threshold( const Point& a, const Point& b, const Point& c )
    {
        return true;
    }
};

struct SupportFn
{
    void operator()( Point& a, Point& b, const Point& dir )
    {

    }
};

int main( int argc, char** argv )
{
    Policy      ops;
    SupportFn   supportFn;

    Eigen::Vector3d a,b,d;
    gjk88::Result result = gjk88::isCollision(ops, supportFn, a, b, d );

    return 0;
}







