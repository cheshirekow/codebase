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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/r2s1/impl/OldKd.cpp
 *
 *  @date   Nov 7, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <mpblocks/kd_tree.hpp>

#include "config.h"
#include "Implementation.h"
#include "impl/ctors.h"


namespace mpblocks {
namespace robot_nn {



struct KdTraits
{
    typedef float Format_t;
    static const unsigned NDim = 3;

    typedef kd_tree::r2_s1::HyperRect<KdTraits> HyperRect;

    class Node:
        public kd_tree::Node<KdTraits>
    {
        public:
            int idx;
    };

};

struct OldKdImplementation:
    public Implementation
{
    kd_tree::Tree<KdTraits>     kd_points;
    std::vector<KdTraits::Node> kd_nodes;
    kd_tree::r2_s1::KNearest<KdTraits> kd_search;

    virtual ~OldKdImplementation(){}

    virtual void allocate( int maxSize )
    {
        kd_nodes.reserve(maxSize*2);
    }

    virtual void insert_derived( int i, const Point& x )
    {
        kd_nodes.push_back( KdTraits::Node() );
        kd_nodes.back().setPoint( x );
        kd_nodes.back().idx  = i;
        kd_points.insert( &kd_nodes.back() );
    }



    virtual void findNearest_derived( const Point& q )
    {
        kd_search.reset(k);
        kd_points.findNearest(q,kd_search);
    }

    virtual void get_result( std::vector<int>& out )
    {
        out.clear();
        out.reserve(k);
        for( auto& item : kd_search.result() )
            out.push_back( item.n->idx );
    }
};


Implementation* impl_old_kd()
{
    return new OldKdImplementation();
}



}
}







