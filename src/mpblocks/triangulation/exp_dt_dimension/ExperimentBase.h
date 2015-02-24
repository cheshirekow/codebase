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
 *  @file   /home/josh/Codes/cpp/mpblocks2/triangulation/src/exp_dt_dimension/ExperimentBase.h
 *
 *  @date   Aug 8, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_EXP_DT_DIMENSION_EXPERIMENTBASE_H_
#define MPBLOCKS_EXP_DT_DIMENSION_EXPERIMENTBASE_H_

#include <vector>
#include <cairomm/cairomm.h>
#include <boost/random.hpp>

namespace         mpblocks {
namespace exp_dt_dimension {


template <typename T>
struct AverageVector:
    std::vector<T>
{
    public:
        typedef std::vector<T> base_t;

    private:
        size_t  m_writeHead;

    public:
        AverageVector():
            m_writeHead(0)
        {}

        void rewind()
        {
            m_writeHead = 0;
        }

        void write( T val )
        {
            assert(m_writeHead < base_t::size());
            (*this)[m_writeHead++] += val;
        }
};


class ExperimentBase
{
    typedef Cairo::RefPtr<Cairo::Context> Cairo_t;
    typedef boost::mt19937                mt19937;

    protected:
        /// the following map iterations to various statistics
        std::string           m_filename;
        AverageVector<int>    m_allSimplices;    ///< num clarkson simplices
        AverageVector<int>    m_hullSimplices;   ///< num simplices in hull
        AverageVector<long>   m_time;            ///< nano seconds since start
        mt19937               m_rng;             ///< random number generator

    public:
        /// is virtual
        virtual ~ExperimentBase(){}

        /// reserve space for a number of iterations
        virtual void initExperiment( int numPoints, int nSimplices )=0;
        void initExperiment( int numPoints );

        /// change the seed and rewind the write head
        void initRun( int seed );

        /// update the number of simplices in the clarkson hull triangulation
        void updateClarksonCount( int nSimplices );

        /// update the number of simplices on the hull surface
        void updateHullCount( int nSimplices );

        /// append a timestamp for the current iteration
        void appendTime( long dt );

        /// plot the clarkson data structure size
        void plotClarksonCount( const Cairo_t& ctx, bool net );

        /// plot the clarkson data structure size
        void plotHullCount( const Cairo_t& ctx, bool net );

        /// plot the clarkson data structure size
        void plotTime( const Cairo_t& ctx, bool net );

        virtual void initialTriangulation()=0;
        virtual void step()=0;

        void save( const std::string& outDir );
};


} //< namespace exp_dt_dimension
} //< namespace mpblocks















#endif // EXPERIMENTBASE_H_
