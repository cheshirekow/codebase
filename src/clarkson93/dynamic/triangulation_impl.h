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
#ifndef CLARKSON93_DYNAMIC_TRIANGULATION_IMPL_H_
#define CLARKSON93_DYNAMIC_TRIANGULATION_IMPL_H_

#include <mpblocks/clarkson93.hpp>

namespace   mpblocks {
namespace clarkson93 {
namespace    dynamic {



template <class Traits>
Triangulation<Traits>::Triangulation():
    m_xv_walked(   simplex::XVISIBLE_WALK ),
    m_x_visible(   simplex::XVISIBLE ),
    m_xv_queue(    simplex::XVISIBLE_QUEUED )
{
    m_ndim = 0;
    clear();
}


template <class Traits>
Triangulation<Traits>::~Triangulation()
{
    clear();
}

template <class Traits>
void Triangulation<Traits>::setup( Setup& setup )
{
    m_antiOrigin    = setup.antiOrigin();
    m_deref         = setup.deref();
    m_ndim          = setup.ndim();

    m_x_visible .reserve( setup.xVisiblePrealloc() );
    m_xv_queue  .reserve( setup.xVisiblePrealloc() );
    m_xv_walked .reserve( setup.xVisiblePrealloc() );
    m_xv_walk   .reserve( setup.xVisiblePrealloc() );

    m_ridges. reserve( setup.horizonPrealloc() );

    setup.setupAlloc( m_sAlloc );
}

template <class Traits>
template <class Iterator>
void Triangulation<Traits>::init(
        Iterator begin, Iterator end)
{
    // just some convenient storage for the initial simplices
    std::vector<Simplex*> S(m_ndim+1);

    // then, if we have enough for the first simplex, we need to build it
    int i=0;
    Simplex* S_0 = m_sAlloc.create();
    for( Iterator ipRef = begin; ipRef != end; ++ipRef )
    {
        S_0->vertices[i++] = ipRef;
    }
    S_0->calculateConstraint( m_ndim, m_deref );
    S_0->orientConstraint( m_deref( S_0->vertices[0] ),
                            SimplexBase_t::INSIDE );


    // we also need to setup all of the infinite simplices
    for(int i=0; i < m_ndim+1; i++)
    {
        S[i] = m_sAlloc.create();
        S[i]->vertices[0]  = m_antiOrigin;
        S[i]->neighbors[0] = S_0;
//        S[i]->inv[0]       = i;
        S_0->neighbors[i]  = S[i];
//        S_0->inv[i]        = 0;

        for(int j=0; j < m_ndim+1; j++)
        {
            int k=j;
            if( j < i )
                k++;
            else if( j==i )
                continue;

            S[i]->vertices[k] = S_0->vertices[j];
        }

        // we can't use the peak vertex of the inifinite simplices to
        // orient them, because it is fictious, so we'll orient it toward
        S[i]->calculateConstraint( m_ndim, m_deref );
        S[i]->orientConstraint( m_deref(S_0->vertices[i]),
                                    SimplexBase_t::OUTSIDE );
    }

    // now we need to assign neighbors
    for(int i=0; i < m_ndim+1; i++)
    {
        // the i'th Simplex, S[i], shares all vertices with S_0 except for
        // S_0's i'th vertex. Also if q is a vertex of S[i] and of S_0, then
        // the neighbor of S_0 opposite q is also the neighbor of S[i]
        // opposite q... I think. I should probably work that out and
        // make sure

        // S[i]'s j'th neighbor is
        //      S_0    if j =  0
        //      S[j-1] if j <= i
        //      S[j]   if j >  i
        // thus the inverse mapping is found by
        //  S[i].N[j]   = S[k]
        //  S[k].N[x]   =
        //      S_0     if x =  0
        //      S[x-1]  if x <= k
        //          => x-1 =  i
        //          => x   =  i+1
        //          => i+1 <= k
        //          => i   <= k-1
        //      S[x]    if x >  k
        //          => x = i
        //          => i > k
        // thus:
        // if( i <= k-1 )
        //      S[k].N[i+1] = S[i]
        // else
        //      S[k].N[i]   = S[i]
        //
        // for example, in 3d:
        //  i = 0, j = 1 => k = 1
        //      S[0].N[1] = S[1]
        //  0 <= 1-1 so i <= k-1 so x=i+1=1 and
        //      S[1].N[1] = S[0]
        //  i =0, j = 2 => k = 2
        //      S[0].N[2] = S[2]
        //  0 <= 2-1 so i <= k-1 so x=i+1=1 and
        //      s[2].N[1] = S[0]
        for(int j=1; j < m_ndim+1; j++)
        {
            int k = (j <= i ) ? j-1 : j;
            S[i]->neighbors[j] = S[k];
//            if( i < k-1 )
//                S[i]->inv[j] = i+1;
//            else
//                S[i]->inv[j] = i;
        }
    }

    m_origin      = S_0;
    m_hullSimplex = S[0];
}


template <class Traits>
void Triangulation<Traits>::insert(PointRef x, Simplex* S)
{
    find_x_visible(x,S);
    alter_x_visible(x);
}



template <class Traits>
void Triangulation<Traits>::clear()
{
    m_hullSimplex = 0;
    m_origin      = 0;
    m_sAlloc.clear();
}

template <class Traits>
void Triangulation<Traits>::find_x_visible(PointRef x, Simplex* S)
{
    // first clear out our search structures
    // so the flags get reset, we do this at the beginning so that after the
    // the update they persist and we can view them for debugging
    m_xv_walk.clear();
    m_xv_walked.clear();

    m_x_visible.clear();
    m_xv_queue.clear();

//  brute force search for x_visible and infinite
//    for( Simplex* S = m_sAlloc.m_start; S != m_sAlloc.m_next; ++S)
//    {
//        if( S->isVisible( m_deref(x) ) && S->isInfinite( m_antiOrigin ) )
//            m_x_visible.push(S);
//    }


    // if we were not given a starting point start at the origin
    if( !S )
        S = m_origin;

    // if the origin simplex is not visible then start at the neighbor
    // across his base, which must be x-visible
    if( !S->isVisible( m_deref(x) ) )
        S = S->neighbors[0];

    // sanity check
    assert( S->isVisible( m_deref(x) ) );

    // star the search at S
    m_xv_walked.push( S );
    m_xv_walk.push( PQ_Key( 0, S ) );

    // clear out S
    S = 0;

    // starting at the given simplex, walk in the direction of x until
    // we find an x-visible infinite simplex (i.e. hull facet)
    while( m_xv_walk.size() > 0 )
    {
        Simplex* pop = m_xv_walk.pop().val;

        // at each step queue all x-visible simplices sorted by origin
        // distance to x
        for(int i=1; i < m_ndim+1; i++)
        {
            Simplex* N = pop->neighbors[i];

            if( N->isVisible( m_deref(x) )
                && !m_xv_walked.isMember(N) )
            {
                // if N is x-visible and infinite then we are done
                if(N->isInfinite(m_antiOrigin) )
                {
                    S = N;
                    break;
                }

                // otherwise add N to the search queue
                Scalar d =
                        ( m_deref(x) - m_deref(N->vertices[0]) ).squaredNorm();
                m_xv_walked.push(N);
                m_xv_walk.push( PQ_Key(d,N) );
            }
        }
    }

    if( !S )
        return;

    // at this S is both x-visible and infinite, so we do an expansive
    // search for all such Simplicies, so we initialize the search stack with
    // this simplex
    m_xv_queue.push(S);

    // at each iteration...
    while( m_xv_queue.size() > 0 )
    {
        // pop one simplex off the stack
        Simplex* S = m_xv_queue.pop();

        // add add it to the x_visible set
        m_x_visible.push(S);

        // and check all of it's neighbors
        for( int i=1; i < m_ndim+1; i++ )
        {
            // if the neighbor is both infinite and x-visible add it to the
            // queue, but don't add it to the queue if it's already determined
            // to be x-visible
            Simplex* N = S->neighbors[i];
            if( N->isVisible( m_deref(x) )
                && N->isInfinite( m_antiOrigin )
                && !m_x_visible.isMember( N )
                && !m_xv_queue.isMember( N ) )
            {
                m_xv_queue.push( N );
            }
        }
    }


    /*
    // start at the origin simplex, and the neighbor sharing it's base
    // facet
    if( !S )
        S = m_origin;
    m_x_visible_queue.push_back(S);
    m_x_visible_queue.push_back(S->neighbors[0]);

    // @todo: there's no reason to keep the whole queue once we get to an
    // infinite simplex. Instead, do a direct line walk through the
    // triangulation toward the point x until we get to an infinite
    // simplex. Then enumerate all infinite and x-visible simplices.

    // enter the dragon... I mean loop
    while(m_x_visible_queue.size() > 0)
    {
        // read the next simplex that needs expansion, remove it from
        // the search queue
        Simplex* simplex = m_x_visible_queue.pop();

        // if it's already been expanded just continue
        if( m_x_visible_expanded.isMember(simplex) )
            continue;

        // otherwise, we'll expand it so add it to the set
        m_x_visible_expanded.push(simplex);

        // check to see if the base facet of this simplex is x-visible,
        // i.e. x is in the same halfspace divided by the base facet as
        // the the rest of the simplex
        if( simplex->isVisible( m_deref(x) ) )
        {
            // if it's also an infinite simplex then add it to the list
            // we're buliding
            if( simplex->isInfinite(m_antiOrigin) )
                m_x_visible.push(simplex);

            // if the base facet is x-visible, then we need to
            // add all of it's neighbors (except for the neighbor
            // across the base facet, from whom we entered this
            // simplex) to the search queue
            for(int i=1; i < m_ndim+1; i++)
            {
                if( !m_x_visible_queue.isMember( simplex->neighbors[i] )
                    && !m_x_visible_expanded.isMember( simplex->neighbors[i] ) )
                {
                    m_x_visible_queue.push( simplex->neighbors[i] );
                }
            }
        }
    }
    */

}



template <class Traits>
void Triangulation<Traits>::alter_x_visible(PointRef x)
{
    // clear out our list of horizon ridges
    m_ridges.clear();

    // first we go through all the x-visible simplices, and replace their
    // peak vertex (the ficitious anti-origin) with the new point x.
    size_t nVisible = m_x_visible.size();
    for(size_t i=0; i < nVisible; ++i)
    {
        Simplex* S = m_x_visible[i];
        S->vertices[0] = x;
    }

    // now we have to find the set of horizon ridges, to do this we
    // find the set of simplices in m_x_visible who have a non base facet G
    // that emits a neighbor who is not in m_x_visible. The intersection of
    // G with the base facet of the simplex in m_x_visible is a horizon
    // ridge
    for(size_t i=0; i < nVisible; ++i)
    {
        Simplex* S = m_x_visible[i];

        // note that we skip index 0, because we are looking for a
        // non-base facet
        for(int i=1; i < m_ndim+1; i++)
        {
            // if the the neighbor is not in x-visible then we have found
            // a non-base facet of this simplex which emits a neighbor
            // which is not in m_x_visible, and so we have found a
            // horizon ridge
            if( !m_x_visible.isMember( S->neighbors[i] ) )
            {
                m_ridges.push(S,i);
                // note that we can't quit, we have to keep searching,
                // there may be more than one horizon ridge
            }
        }
    }

    // now for each horizon ridge we have to construct a new simplex
    size_t nHorizon = m_ridges.size();
    for( size_t i=0; i < nHorizon; i++ )
    {
        HorizonRidge_t& ridge = m_ridges[i];

        // allocate a new simplex and add it to the list
        Simplex* simplex = m_sAlloc.create();
        ridge.generatedSimplex = simplex;

        // In the parlance of Clarkson, we have two simplices V and N
        // note that V is ridge->simplex and N is neighbor
        // ridge->simplex->neighbors[i]. V was an infinite simplex and became
        // a finite one... and N is still an infinite simplex
        Simplex* V = ridge.simplex;
        Simplex* N = V->neighbors[ridge.iExcluded];

        // the new simplex has a peak vertex at the anti-origin, across from
        // that vertex is V. Furthermore, V->neighbors[ridge.iExcluded] will
        // move to point to the new simplex
        simplex->vertices[0]          = m_antiOrigin;
        simplex->neighbors[0]         = V;
//        simplex->inv[0]               = ridge.iExcluded;

        // note that the neighbor of V opposite it's peak has already been
        // set, in
        // fact the only neighbor that needs to be set is the neighbor
        // across from iExclude which will be the new simplex, so lets
        // set these guys now, for simplicity, we'll use the same index
        // in the new simplex
        simplex->vertices[ridge.iExcluded]     = x;
        simplex->neighbors[ridge.iExcluded]    = N;
//        simplex->inv[ridge.iExcluded]          = V->inv[ridge.iExcluded];

        // before we assign V's iExcluded neighbor to the new simplex,
        // we need to also assign THAT simplex's neighbor to
        // this new simplex
        {
            // so we scan the neighbors of i to find the index which is
            // V, and once we find that index, we set that neighbor to the
            // new simplex
            // @todo replace this loop with V->inv[ridge.iExcluded]
            for(int i=0; i < m_ndim+1; i++)
            {
                if( N->neighbors[i] == V )
                {
                    N->neighbors[i] = simplex;
//                    N->inv[i]       = ridge.iExcluded;
                    break;
                }
            }
        }

        V->neighbors[ridge.iExcluded] = simplex;
//        V->      inv[ridge.iExcluded] = 0;

        // otherwise it shares all non-base vertices with that of the
        // simplex which generated this horizon ridge (excluding the
        // one whose index we recorded )
        for(int i = 1; i < m_ndim+1; i++)
        {
            // if this is the index of the excluded vertex, then we've
            // already done the work so move on
            if( i == ridge.iExcluded )
                continue;

            // otherwise we copy the vertex from the originating
            // simplex
            else
            {
                simplex->vertices[i]  = V->vertices[i];
                // note that we'll set the new simplices neighbors in
                // another loop
            }
        }

        // we'll need to use the half-space inequality at some point but
        // we can't calcualte it normally b/c one of the vertices isn't at
        // a real location, so we calculate it to be coincident to the base
        // facet, and then orient it so that the vertex of V which is not
        // part of the new simplex is on the other side of the constraint
        PointRef p_excluded = V->vertices[ridge.iExcluded];
        simplex->calculateConstraint( m_ndim, m_deref );
        simplex->orientConstraint( m_deref(p_excluded),
                                    SimplexBase_t::OUTSIDE);
    }


    // ok now that all the new simplices have been added, we need to go
    // and assign neighbors to these new simplicies
    for( size_t i=0; i < nHorizon; i++ )
    {
        HorizonRidge_t& ridge = m_ridges[i];
        Simplex* simplex = ridge.generatedSimplex;
        Simplex* V       = ridge.simplex;

        // set of vertices of the new simplex, note that we exclude i=0
        // which is the anti-origin, and iExclude, which is x
        PointSet vertices_f;
        for(int i=1; i < m_ndim+1; i++)
            vertices_f.insert( simplex->vertices[i] );

        // we need to find all of the neighbors except for the iExcluded
        // one and for the one across the base facet
        for(int i=1; i < m_ndim+1; i++)
        {
            if( i == ridge.iExcluded )
                continue;

            PointRef  q       = simplex->vertices[i];
            Simplex*  from    = 0;
            Simplex*  next    = V;
            Simplex*  S       = 0;

            // first, we remove q from vertices(f), note that there are
            // now d-1 points in this set, defining a d-2 face (simplex?)
            vertices_f.erase(q);

            // It is guarenteed that V is in the set script S. So we start
            // at V and find it's neighbor who is also in script S
            // (if it exists). Then we continue a walk in this manner,
            // because each element of script S has at most two
            // neighbors in script S.
            while(next != S)
            {
                from = S;
                S    = next;
                for(int j=0; j < m_ndim+1; j++)
                {
                    Simplex* candidate = S->neighbors[j];
                    if( candidate != from
                            && candidate->vertices[0] == x)
                    {
                        // in order to be in Script S, other than x, it can
                        // have at most 2 vertex not in vertices_f - q,
                        int nFail = 0;
                        for(int k=1; k < m_ndim+1; k++)
                        {
                            if( vertices_f.find(candidate->vertices[k])
                                    == vertices_f.end() )
                            {
                                nFail++;
                                if(nFail > 2)
                                    break;
                            }
                        }

                        if(nFail < 3)
                        {
                            // if candidate is not the neighbor we entered from,
                            // and has x as a peak, and contains only one vertex
                            // not in vertices(f) \ q, then it must be a member
                            // of Script S.
                            next = candidate;
                            break;
                        }
                    }
                }

            }

            // now that we're here, the base facet of S must be composed
            // of vertices_f \ q U {y1, y2}. Furthemore the neighbor
            // opposite y2 must be the neighbor that we entered S by
            // (or q in the the case of a zero length path). So we need
            // to identify which of the vertices is y1, which will give us
            // the neighbor of the new simplex opposite q
            int iy1 = 0;

            // first check the case that V = V'
            if( S == V )
            {
                for(int j=1; j < m_ndim+1; j++)
                {
                    Simplex* neighbor = S->neighbors[j];
                    PointRef vertex   = S->vertices[j];

                    if(vertex == q)
                    {
                        iy1 = j;
                        break;
                    }
                }
            }

            else
            {
                // note that we don't check j=0 because we're interested only
                // in the vertices on the base
                for(int j=1; j < m_ndim+1; j++)
                {
                    Simplex* neighbor = S->neighbors[j];
                    PointRef vertex   = S->vertices[j];

                    // this vertex is y2 then just keep searching
                    if( neighbor == from )
                        continue;

                    // otherwise, check to see if this vertex is not in
                    // vertices_f
                    if( vertices_f.find(vertex) == vertices_f.end() )
                    {
                        iy1 = j;
                        break;
                    }
                }
            }

            // ok, now that we've found iy2, it shoudl be that the i'th
            // neighbor of the new simplex, i.e. the neighbor opposite
            // q is the neighbor opposite the y2 of S.
            simplex->neighbors[i] = S->neighbors[iy1];

            // now we put q back into vertices(f)
            vertices_f.insert(q);
        }
    }
}





} // namespace dynamic
} // namespace clarkson93
} // namespace mpblocks





#endif  // CLARKSON93_DYNAMIC_TRIANGULATION_IMPL_H_
