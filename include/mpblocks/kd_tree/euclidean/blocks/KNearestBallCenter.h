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
 *  @file mpblocks/kd_tree/euclidean/blocks/KNearestBallCenter.h
 *
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_EUCLIDEAN_BLOCKS_KNEARESTBALLCENTER_H_
#define MPBLOCKS_KD_TREE_EUCLIDEAN_BLOCKS_KNEARESTBALLCENTER_H_

#include <set>

namespace  mpblocks {
namespace   kd_tree {
namespace euclidean {
namespace    blocks {

/// performs search for k-NN within a ball at a specified center and radius
template <class Traits,
            template<class> class Allocator = std::allocator >
class KNearestBallCenter
{
    public:
        typedef typename Traits::Format_t    Format_t;
        typedef typename Traits::Node        Node_t;

        typedef Eigen::Matrix<Format_t,Traits::NDim,1>  Point_t;

        typedef KNearestBallCenter<Traits,Allocator>              This_t;
        typedef euclidean::KNearestBallCenter<Traits,Allocator>   Search_t;
        typedef Tree<Traits>                                Tree_t;
        typedef typename Search_t::PQueue_t                 PQueue_t;

    protected:
        Tree_t*  m_tree;
        Search_t m_search;

    public:
        sigc::signal<void,Node_t*>  sig_result;

        KNearestBallCenter(Tree_t* tree=0):
            m_tree(tree)
        {}

        void setTree( Tree_t* tree )
            { m_tree = tree; }

        void search( const Point_t& q, const Point_t& c,
                    Format_t r, unsigned int k )
        {
            m_search.reset(c,r,k);
            if(m_tree)
            {
                m_tree->findNearest(q,m_search);
                const PQueue_t& list = m_search.result();
                typename PQueue_t::const_iterator iKey;
                for( iKey = list.begin(); iKey != list.end(); ++iKey )
                {
                    sig_result.emit(iKey->n);
                }
            }
        }

        sigc::slot<void,const Point_t&,const Point_t&,Format_t,unsigned int>
        slot_search()
            { return sigc::mem_fun(this,&This_t::search); }
};


} // namespace blocks
} // namespace euclidean
} // namespace kd_tree
} // namespace mpblocks

#endif
