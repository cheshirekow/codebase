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
 *  @file   /home/josh/Codes/cpp/mpblocks2/triangulation/src/exp_dt_dimension/ExperimentBase.cpp
 *
 *  @date   Aug 8, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */


#include "ExperimentBase.h"
#include <fstream>


namespace         mpblocks {
namespace exp_dt_dimension {

struct ClearContainer
{
    template <typename T>
    void operator()( T& container ) const
    {
        container.clear();
    }
};

struct ReserveContainer
{
    size_t m_size;

    ReserveContainer( size_t size ):
        m_size(size)
    {}

    template <typename T>
    void operator()( T& container ) const
    {
        container.resize( m_size, 0 );
    }
};

struct Rewind
{
    template <typename T>
    void operator()( T& container ) const
    {
        container.rewind();
    }
};



template <typename T>
void plot( const Cairo::RefPtr<Cairo::Context>& ctx, const T& container, bool accum )
{
    double i     = 0;
    double value = 0;

    ctx->move_to(0,0);
    for( auto val : container )
    {
        if( accum )
            value += val;
        else
            value = val;
        ctx->line_to(++i,value);
    }
}


template <typename Op>
void applyOp( const Op& op){}

/// apply an operation to every item in a list
template <typename Op, typename Head, typename... Tail>
void applyOp( const Op& op, Head& head, Tail&... tail )
{
    op(head);
    applyOp(op,tail...);
}



void ExperimentBase::initExperiment( int numPoints )
{
    applyOp( ClearContainer(),
                m_allSimplices,
                m_hullSimplices,
                m_time );

    applyOp( ReserveContainer(numPoints),
                m_allSimplices,
                m_hullSimplices,
                m_time );
}

void ExperimentBase::initRun( int seed )
{
    m_rng.seed(seed);
    applyOp( Rewind(),
                m_allSimplices,
                m_hullSimplices,
                m_time );
    initialTriangulation();
}

void ExperimentBase::updateClarksonCount( int nSimplices )
{
    m_allSimplices.write(nSimplices);
}

void ExperimentBase::updateHullCount( int nSimplices )
{
    m_hullSimplices.write(nSimplices);
}

void ExperimentBase::appendTime( long dt )
{
    m_time.write(dt);
}

void ExperimentBase::plotClarksonCount( const Cairo_t& ctx, bool net )
{
    plot( ctx, m_allSimplices, net );
}

void ExperimentBase::plotHullCount( const Cairo_t& ctx, bool net )
{
    plot( ctx, m_hullSimplices, net );
}

void ExperimentBase::plotTime( const Cairo_t& ctx, bool net )
{
    plot( ctx, m_time, net );
}

void ExperimentBase::save( const std::string& outDir )
{
    std::ofstream out( outDir + "/" + m_filename );

    size_t n = m_time.size();
    for( size_t i = 0; i < n ; i ++ )
    {
        out << i << ","
            << m_time[i] << ","
            << m_allSimplices[i] << ","
            << m_hullSimplices[i] << "\n";
    }

    out.close();
}




} //< namespace exp_dt_dimension
} //< namespace mpblocks






