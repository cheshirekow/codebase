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
 *  @file   src/Socket.cpp
 *
 *  @date   Jan 6, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <cpp_fcgi/Socket.h>
#include <cstdio>
#include <unistd.h>

namespace fcgi {

int Socket::open( const char* path, int backlog )
{
    m_fd = FCGX_OpenSocket( path, backlog );
    return m_fd;
}

int Socket::open( int port, int backlog )
{
    char buf[10];
    snprintf( buf, 10, ":%d", port );
    m_fd = FCGX_OpenSocket( buf, backlog );
    return m_fd;
}

int Socket::close()
{
    return ::close(m_fd);
}

int Socket::fd() const
{
    return m_fd;
}

}







