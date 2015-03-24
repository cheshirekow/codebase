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
 *  @file   mpblocks/kd_tree/r2_s1/blocks/Nearest.h
 *
 *  @date   Nov 20, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_KD_TREE_R2_S1_BLOCKS_NEAREST_H_
#define MPBLOCKS_KD_TREE_R2_S1_BLOCKS_NEAREST_H_


namespace mpblocks {
namespace  kd_tree {
namespace    r2_s1 {
namespace   blocks {

template <class Traits>
class Nearest
{
    public:
        typedef typename Traits::Format_t   Format_t;
        typedef typename Traits::Node       Node_t;

        typedef Eigen::Matrix<Format_t,Traits::NDim,1>  Point_t;

        typedef Nearest<Traits>          This_t;
        typedef r2_s1::Nearest<Traits>   Search_t;
        typedef Tree<Traits>             Tree_t;

    protected:
        Tree_t*  m_tree;
        Search_t m_search;

    public:
        sigc::signal<void,Node_t*>  sig_result;

        Nearest(Tree_t* tree=0):
            m_tree(tree)
        {}

        void setTree( Tree_t* tree )
            { m_tree = tree; }

        void search( const Point_t& q )
        {
            m_search.reset();
            if(m_tree)
            {
                m_tree->findNearest(q,m_search);
                sig_result.emit(m_search.result());
            }
        }

        sigc::slot<void,const Point_t&> slot_search()
            { return sigc::mem_fun(this,&This_t::search); }
};





} // namespace blocks
} // namespace r2_s1
} // namespace kd_tree
} // namespace mpblocks








#endif // NEARESTNEIGHBOR_H_
