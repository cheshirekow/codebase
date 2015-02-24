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
 *  @file   /home/josh/Codes/cpp/mpblocks2/robot_nn/src/binarytree.h
 *
 *  @date   Nov 5, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_ROBOT_NN_BINARYTREE_H_
#define MPBLOCKS_ROBOT_NN_BINARYTREE_H_

#include <map>

namespace mpblocks {
namespace robot_nn {
namespace      bst {

template <typename Scalar>
struct Node
{
    Scalar  val;
    int     pointRef;
    Node*   next;
    Node*   prev;

    Node(Scalar val = Scalar(0.0), int pointRef=-1 ):
        val(val),
        pointRef(pointRef),
        next(this),
        prev(this)
    {}

    struct LessThan
    {
        bool operator()( const Node& a, const Node& b )
        {
            if( a.val < b.val )
                return true;
            else if( a.val > b.val )
                return false;
            else
                return a.pointRef < b.pointRef;
        }
    };
};

template <typename Scalar>
struct Cell
{
    Node<Scalar>* first;
    Node<Scalar>* last;
};

template <typename Scalar>
struct SetList
{
    typedef std::set< Node<Scalar> > NodeSet;
    NodeSet set;

    Cell<Scalar>  getCell( Scalar val )
    {
        typename NodeSet::iterator iter =
            set.lower_bound( Node<Scalar>(val,0) );
        if( iter == set.end() )
            iter = set.begin();
        Node<Scalar>* last = &(*iter);

        return Cell<Scalar>( last->prev, last );
    }

    Node<Scalar>* insert( int pointRef, Scalar val )
    {
        auto pair = set.insert( Node<Scalar>(val,pointRef) );
        typename NodeSet::iterator nextIter = pair.second;

        nextIter++;
        if( nextIter == set.end() )
            nextIter = set.begin();

        Node<Scalar>* node = &(*pair.second);
        Node<Scalar>* next = &(*nextIter);
        Node<Scalar>* prev = next->prev;

        next->prev = node;
        prev->next = node;
        node->prev = prev;
        node->next = next;

        return node;
    }

};




} //< bst
} //< robot_nn
} //< mpblocks














#endif // BINARYTREE_H_
