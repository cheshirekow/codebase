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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/r2s1/impl/kd2.h
 *
 *  @date   Nov 7, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_KD2_H_
#define MPBLOCKS_KD2_H_

#include "config.h"
#include "Implementation.h"
#include "impl/ctors.h"
#include <mpblocks/dubins/curves_eigen.hpp>

namespace mpblocks {
namespace robot_nn {
namespace   dubins {

/// regular quad trees
struct BruteForceImplementation:
    public Implementation
{
    typedef Eigen::Matrix<float,3,1> Point;

    struct ResultKey
    {
        float dist;
        int   point;

        ResultKey():
            dist(0),
            point(0)
        {}

        ResultKey( float dist, int point ):
            dist(dist),
            point(point)
        {}

        struct LessThan
        {
            bool operator()( const ResultKey& a, const ResultKey& b )
            {
                if( a.dist < b.dist )
                    return true;
                else if( a.dist > b.dist )
                    return false;
                else
                    return a.point < b.point;
            }
        };
    };

    std::vector<Point>&     points;
    std::vector<ResultKey>  result; //< max heap
    ResultKey::LessThan     compare;
    int                     size;

    BruteForceImplementation( std::vector<Point>& points):
        points(points)
    {
        result.reserve(k+1);
        size = 0;
    }

    virtual ~BruteForceImplementation(){}

    virtual void allocate( int maxSize ){}

    virtual void insert_derived( int i, const Point& x )
    {
        size++;
    }

    virtual void findNearest_derived( const Point& q )
    {
        result.clear();

        for( int i=0; i < size; i++ )
        {
            const Point& x = points[i];

            mpblocks::dubins::curves_eigen::Path<float> dubins_result =
                mpblocks::dubins::curves_eigen::solve(q, x, m_r );

            ResultKey key( dubins_result.dist(m_r), i );
            result.push_back(key);
            std::push_heap( result.begin(), result.end(), compare );
            if( result.size() > k )
            {
                std::pop_heap( result.begin(), result.end(), compare );
                result.pop_back();
            }
        }
    }

    virtual void get_result( std::vector<int>& out )
    {
        out.clear();
        out.reserve(k);
        for(auto& item : result )
            out.push_back( item.point );
    }
};


Implementation* impl_cpu_bf(std::vector<Point>& points )
{
    return new BruteForceImplementation(points);
}

}
}
}



#endif // KD2_H_
