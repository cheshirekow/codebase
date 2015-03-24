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
 *  @file
 *
 *  @date   Mar 26, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_KD_TREE_NODE_HPP_
#define MPBLOCKS_KD_TREE_NODE_HPP_

#include <limits>


namespace mpblocks {
namespace  kd_tree {


template <class Traits>
Node<Traits>::Node()
{
    m_parent        = 0;
    m_i             = 0;
    m_smallerChild  = 0;
    m_greaterChild  = 0;
}

template <class Traits>
void Node<Traits>::construct(
        Node_t* parent, unsigned int  i )
{
    m_parent        = parent;
    m_i             = i;
    m_smallerChild  = 0;
    m_greaterChild  = 0;
    m_this = static_cast<Node_t*>(this);
}

template <class Traits>
void Node<Traits>::setPoint( const Point_t& p )
{
    m_point = p;
}

template <class Traits>
const typename Node<Traits>::Point_t&
    Node<Traits>::getPoint()
{
    return m_point;
}

template <class Traits>
void Node<Traits>::insert( Node_t* node )
{
    Node_t** ptrChild = 0;

    // first, grab a pointer to which child pointer we should recurse
    // into
    if(static_cast<This_t*>(node)->m_point[m_i] <= m_point[m_i])
        ptrChild = &m_smallerChild;
    else
        ptrChild = &m_greaterChild;

    // dereference the pointer
    Node_t*& child = *ptrChild;

    // if the child exists (is not null) then recurse, otherwise
    // create it and we're done
    if(child)
        return static_cast<This_t*>(child)->insert(node);
    else
    {
        child = node;
        static_cast<This_t*>(node)->construct( m_this,(m_i+1) % m_point.rows());
    }
}



template <class Traits>
void Node<Traits>::findNearest(
        const Point_t&  q,
        HyperRect_t&    rect,
        NNIface_t&      search)
{
    Node_t*     nearerNode;
    Node_t*     fartherNode;
    Format_t*   nearerHyperCoord;
    Format_t*   fartherHyperCoord;

    // first, check to see if the left child or the right child is a
    // would-be-ancester if the point were incerted in the graph
    Format_t  diff = q[m_i] - m_point[m_i];
    if (diff <= 0.0)
    {
        nearerNode  = m_smallerChild;
        fartherNode = m_greaterChild;

        nearerHyperCoord    = &(rect.maxExt[m_i]);
        fartherHyperCoord   = &(rect.minExt[m_i]);
    }

    else
    {
        nearerNode  = m_greaterChild;
        fartherNode = m_smallerChild;

        nearerHyperCoord    = &(rect.minExt[m_i]);
        fartherHyperCoord   = &(rect.maxExt[m_i]);
    }

    // now, whichever child is the would-be-ancestor, recurse into them
    // also, refine the hyperrectangle that contains the node by
    // splitting it along the split-plane of this node
    if (nearerNode)
    {
        // copy out the old extent of they hyper-rectangle
        Format_t oldHyperVal = *nearerHyperCoord;

        // split the hyperrectangle by updating the extent with the
        // value of this nodes split plane
        *nearerHyperCoord = m_point[m_i];

        // recurse into the nearer node
        static_cast<This_t*>(nearerNode)->findNearest(q,rect,search);

        // now that we've stepped back up into this node, restore the
        // hyperrectangle
        *nearerHyperCoord = oldHyperVal;
    }

    // evaluate this node and add it to the result set if necessary
    search.evaluate(q,m_point,m_this);

    // if the farther node exists, we might need to also check it's
    // children
    if (fartherNode)
    {
        // refine the hyper-rectangle of the farther subtree
        Format_t oldHyperVal = *fartherHyperCoord;
        *fartherHyperCoord = m_point[m_i];

        // check if we have to recurse into the farther subtree
        if( search.shouldRecurse(q,rect) )
            static_cast<This_t*>(fartherNode)->findNearest(q,rect,search);

        // undo refinement of the hyperrectangle
        *fartherHyperCoord = oldHyperVal;
    }
}







template <class Traits>
void Node<Traits>::findRange(RangeIface_t& search,HyperRect_t& rect)
{
    // evaluate this node and add it to the result set if
    // it lies inside the range
    search.evaluate(m_point,m_this);

    // now evaluate the two children and recurse if necessary
    {
        // copy out the old extent of they hyper-rectangle
        Node_t*   child       = m_smallerChild;
        Format_t* hyperCoord  = &(rect.maxExt[m_i]);
        Format_t  oldHyperVal = *hyperCoord;

        // split the hyperrectangle by updating the extent with the
        // value of this nodes split plane
        *hyperCoord = m_point[m_i];

        // recurse into the nearer node
        if(child && search.shouldRecurse(rect) )
            static_cast<This_t*>(child)->findRange(search,rect);

        // now that we've stepped back up into this node, restore the
        // hyperrectangle
        *hyperCoord = oldHyperVal;
    }

    {
        // copy out the old extent of they hyper-rectangle
        Node_t*   child       = m_greaterChild;
        Format_t* hyperCoord  = &(rect.minExt[m_i]);
        Format_t  oldHyperVal = *hyperCoord;

        // split the hyperrectangle by updating the extent with the
        // value of this nodes split plane
        *hyperCoord = m_point[m_i];

        // recurse into the nearer node
        if(child && search.shouldRecurse(rect) )
            static_cast<This_t*>(child)->findRange(search,rect);

        // now that we've stepped back up into this node, restore the
        // hyperrectangle
        *hyperCoord = oldHyperVal;
    }
}


template <class Traits>
template <typename BackInserter >
void Node<Traits>::enumerate(
        HyperRect_t& container, BackInserter bs )
{
    if( m_greaterChild )
    {
        Pair_t* pair= new Pair_t();
        container.copyTo(pair->container);
        pair->container.minExt[m_i] = m_point[m_i];
        pair->node = m_greaterChild;
        bs = pair;
    }

    if( m_smallerChild )
    {
        Pair_t* pair= new Pair_t();
        container.copyTo(pair->container);
        pair->container.maxExt[m_i] = m_point[m_i];
        pair->node = m_smallerChild;
        bs = pair;
    }
}










} // namespace kd_tree
} // mpblocks

#endif
