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
 *  @file   /home/josh/Codes/cpp/mpblocks2/triangulation/src/exp_dt_dimension/Runner.h
 *
 *  @date   Aug 8, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_EXP_DT_DIMENSION_RUNNER_H_
#define MPBLOCKS_EXP_DT_DIMENSION_RUNNER_H_

#include "ExperimentBase.h"

#include <cpp_pthreads.h>

namespace         mpblocks {
namespace exp_dt_dimension {


struct Runner
{
    enum Key{
        KEY_STEPS,
        KEY_TRIALS,
        KEY_BATCH,
        KEY_SIMPLICES,
    };

    private:
        ExperimentBase*     m_exp;

        pthreads::Thread    m_thread;
        pthreads::Mutex     m_mutex;

        unsigned int        m_step;
        unsigned int        m_numSteps;
        unsigned int        m_trial;
        unsigned int        m_numTrials;
        unsigned int        m_batchSize;

        unsigned int        m_reserveSimplices;

        volatile bool    m_needsInit;
        volatile bool    m_shouldQuit;
        volatile bool    m_isRunning;
        volatile bool    m_shouldJoin;

        double  m_trialProgress;
        double  m_totalProgress;

    public:
        Runner( ExperimentBase* exp=0 );
        ~Runner();
        void setExperiment( ExperimentBase* exp );
        void quit();
        void join();
        double trialProgress();
        double totalProgress();
        void setSteps( int numSteps );
        void setTrials( int numTrials );
        void setBatch( int numBatch );
        void setSimplices( int numSimplices );
        void set( Key key, int val );
        void reset();
        void launch();
        void operator()();
};



} //< namespace exp_dt_dimension
} //< namespace mpblocks





#endif // RUNNER_H_
