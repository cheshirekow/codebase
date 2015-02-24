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
 *  @file   /home/josh/Codes/cpp/mpblocks2/triangulation/src/exp_dt_dimension/Runner.cpp
 *
 *  @date   Aug 8, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include "Runner.h"



namespace         mpblocks {
namespace exp_dt_dimension {


Runner::Runner( ExperimentBase* exp ):
    m_exp(exp),
    m_step(0),
    m_numSteps(100),
    m_trial(0),
    m_numTrials(10),
    m_batchSize(10),
    m_reserveSimplices(100),
    m_needsInit(true),
    m_shouldQuit(false),
    m_isRunning(false),
    m_shouldJoin(false),
    m_trialProgress(0),
    m_totalProgress(0)
{
    m_mutex.init();
}

Runner::~Runner()
{
    m_mutex.destroy();
}

void Runner::setExperiment( ExperimentBase* exp )
{
    m_exp = exp;
}

void Runner::quit()
{
    pthreads::ScopedLock lock( m_mutex );
    m_shouldQuit = true;
}

void Runner::join()
{
    pthreads::ScopedLock lock( m_mutex );
    while( m_isRunning )
    {
        // release the lock, then immediately reaquire it, should give
        // the thread time to quit if it is waiting for the lock
        m_mutex.unlock();
        m_mutex.lock();
    }

    // release thread memory
    if( m_shouldJoin )
        m_thread.join();
}

double Runner::trialProgress()
{
    pthreads::ScopedLock lock( m_mutex );
    return m_trialProgress;
}

double Runner::totalProgress()
{
    pthreads::ScopedLock lock( m_mutex );
    return m_totalProgress;
}

void Runner::setSteps( int numSteps )
{
    pthreads::ScopedLock lock( m_mutex );
    m_numSteps  = numSteps;
    m_needsInit = true;
}

void Runner::setTrials( int numTrials )
{
    pthreads::ScopedLock lock( m_mutex );
    m_numTrials = numTrials;
}

void Runner::setBatch( int numBatch )
{
    pthreads::ScopedLock lock( m_mutex );
    m_batchSize = numBatch;
}

void Runner::setSimplices( int numSimplices )
{
    pthreads::ScopedLock lock( m_mutex );
    m_reserveSimplices = numSimplices;
    m_needsInit        = true;
}

void Runner::set( Key key, int val )
{
    pthreads::ScopedLock lock( m_mutex );
    switch(key)
    {
        case KEY_STEPS:
            m_numSteps  = val;
            m_needsInit = true;
            break;

        case KEY_TRIALS:
            m_numTrials = val;
            break;

        case KEY_BATCH:
            m_batchSize = val;
            break;

        case KEY_SIMPLICES:
            m_reserveSimplices = val;
            m_needsInit        = true;
            break;

        default:
            break;
    }
}

void Runner::reset()
{
    pthreads::ScopedLock lock( m_mutex );
    m_needsInit = true;
}

void Runner::launch()
{
    pthreads::ScopedLock lock( m_mutex );
    if( !m_isRunning )
    {
        m_shouldQuit = false;
        m_isRunning  = true;
        m_thread.launch(this);
    }
}

void Runner::operator()()
{
    bool shouldQuit = false;
    unsigned int  numTrials  = 10;
    unsigned int  numSteps   = 10;
    unsigned int  batchSize  = 10;
    unsigned int  reserveSimplices = 100;
    bool          needsInit        = false;

    unsigned int  step  = 0;
    unsigned int  trial = 0;

    std::cout << "Thread is starting up: " << pthreads::Thread::self().c_obj()
              << "\n";

    // restore previous state
    {
        pthreads::ScopedLock lock( m_mutex );
        numTrials = m_numTrials;
        numSteps  = m_numSteps;
        batchSize = m_batchSize;
        reserveSimplices = m_reserveSimplices;

        // since we've captured needsInit, it will be handled before
        // exiting and we can flip the flag
        needsInit        = m_needsInit;

        step  = m_step;
        trial = m_trial;
    }

    std::cout << "Thread " << pthreads::Thread::self().c_obj()
              << " entering main loop\n";

    while( !shouldQuit )
    {
        if(needsInit)
        {
            std::cout << "Thread " << pthreads::Thread::self().c_obj()
              << " initializing experiment\n";
            m_exp->initExperiment(numSteps,reserveSimplices);
            step  = 0;
            trial = 0;
            needsInit = false;

            pthreads::ScopedLock lock( m_mutex );
            m_needsInit = false;
        }

        for( ; trial < numTrials && !shouldQuit && !needsInit; trial++ )
        {
//            std::cout << "Thread " << pthreads::Thread::self().c_obj()
//              << " initializing trial\n";
            // only init the run if we are not continuing after pause
            if( step == 0 )
                m_exp->initRun(trial);

//            std::cout << "Thread " << pthreads::Thread::self().c_obj()
//              << " starting trial\n";

            for( ; step < numSteps; step+= batchSize )
            {
                int n = (step + batchSize < numSteps) ?
                            batchSize : numSteps - step;

//                std::cout << "Thread " << pthreads::Thread::self().c_obj()
//                      << " starting batch\n";

                // do a batch of iterations
                for(int i=0; i < n; i++)
                    m_exp->step();

//                std::cout << "Thread " << pthreads::Thread::self().c_obj()
//                      << " completed batch\n";

                // now syncrhonize with other thread, we only accept updates to
                // numTrials or batchSize since other updates would require an
                // initialization
                pthreads::ScopedLock lock( m_mutex );
                needsInit  = m_needsInit;
                shouldQuit = m_shouldQuit;
                numTrials  = m_numTrials;
                batchSize  = m_batchSize;

                // we also record our current status so that the other thread
                // can read it
                m_trialProgress = (step+1) / (double)numSteps;
                m_totalProgress = (trial + m_trialProgress) / (double)numTrials;

                if(needsInit || shouldQuit)
                    break;
            }

            // if we finished the trial then reset
            if( !shouldQuit )
                step = 0;
        }

        if(trial == numTrials)
            break;
    }

    std::cout << "Thread is shutting down: " << pthreads::Thread::self().c_obj()
              << "\n";

    // before shutting down we record our state so that we can resume later
    pthreads::ScopedLock lock( m_mutex );
    m_step  = step;
    m_trial = trial;
    m_isRunning  = false;
    m_shouldJoin = true;
}




} //< namespace exp_dt_dimension
} //< namespace mpblocks








