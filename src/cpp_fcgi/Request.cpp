/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of cpp-pthreads.
 *
 *  cpp-pthreads is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  cpp-pthreads is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with cpp-pthreads.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   src/Request.cpp
 *
 *  @date   Jan 6, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <cpp_fcgi/Request.h>


namespace fcgi {

Request::Request():
    m_in(0),
    m_out(0),
    m_err(0)
{}

int Request::init( const Socket& sock, int flags )
{
    int retVal = FCGX_InitRequest( &m_req, sock.fd(), flags );

    if( retVal == 0)
    {
        m_in.rdbuf(&m_inBuf);
        m_out.rdbuf(&m_outBuf);
        m_err.rdbuf(&m_errBuf);
    }

    return retVal;
}

int Request::accept()
{
    int retVal = FCGX_Accept_r(&m_req);
    if( retVal==0 )
    {
        m_inBuf.attach(m_req.in);
        m_outBuf.attach(m_req.out);
        m_errBuf.attach(m_req.err);
    }
    return retVal;
}

void Request::finish()
{
    FCGX_Finish_r(&m_req);
}

void Request::free()
{
    FCGX_Free( &m_req, 0 );
}

const char* Request::getParam( const char* name )
{
    return FCGX_GetParam( name, m_req.envp);
}

char** Request::envp()
{
    return m_req.envp;
}

std::istream& Request::in()
{
    return m_in;
}

std::ostream& Request::out()
{
    return m_out;
}

std::ostream& Request::err()
{
    return m_err;
}



}







