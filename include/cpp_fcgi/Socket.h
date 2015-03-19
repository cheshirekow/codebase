/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of cpp-fcgi.
 *
 *  cpp-fcgi is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  cpp-fcgi is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with cpp-fcgi.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   include/cpp-fcgi/Socket.h
 *
 *  @date   Jan 6, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  Defines Socket class
 */

#ifndef CPP_FCGI_SOCKET_H_
#define CPP_FCGI_SOCKET_H_

#include <fcgiapp.h>

namespace fcgi {

/// An fcgi socket connection
class Socket
{
    private:
        int m_fd;   ///< file descriptor of the open socket

    public:
        /// open an fcgi socket on a unix domain
        /**
         *  @param  path    path of the unix domain socket
         *  @param  backlog number of unprocessed requests to queue
         *  @return the fd of the opened socket or an error number
         */
        int open( const char* path, int backlog );

        /// open an fcgi socket listening on the specified port
        /**
         *  @param  port    port to listen on
         *  @param  backlog number of unprocessed requests to queue
         *  @return the fd of the opened socket or an error number
         */
        int open( int port, int backlog);

        /// close the socket
        /**
         *  @return 0 on success or an error number
         */
        int close();

        /// return the file descriptor
        int fd() const;
};


}






#endif // SOCKET_H_
