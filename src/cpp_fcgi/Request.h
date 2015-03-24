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
 *  @file   include/cpp-fcgi/Request.h
 *
 *  @date   Jan 6, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  Defines Request class
 */

#ifndef CPP_FCGI_REQUEST_H_
#define CPP_FCGI_REQUEST_H_

#include <fcgiapp.h>
#include <fcgio.h>
#include <iostream>
#include <cpp_fcgi/Socket.h>

namespace fcgi {

/// An FCGI request object
class Request
{
    private:
        FCGX_Request    m_req;      ///< the actual request object

        fcgi_streambuf  m_inBuf;    ///< std::streambuf for the input stream
        fcgi_streambuf  m_outBuf;   ///< std::streambuf for the output stream
        fcgi_streambuf  m_errBuf;   ///< std::streambuf for the error stream

        std::istream    m_in;   ///< input stream
        std::ostream    m_out;  ///< output stream
        std::ostream    m_err;  ///< error stream

    public:
        Request();

        /// Initialize an FCGX_Request for use with accept_r
        /**
         *  @param sock     the socket connection to accept a request on
         *  @param flags    for now, only accepts FCGI_ACCEPT_ON_INTR
         *  @return result of FCGX_InitRequest
         *
         *  @note   Most methods are not valid until accept() has been called
         *          successfully
         */
        int init( const Socket& sock, int flags=FCGI_FAIL_ACCEPT_ON_INTR );

        /// accepts a new request
        /**
         *  @return result of FCGX_accept_r
         */
        int accept();

        /// frees memory from a serviced request
        /**
         *  @note   After calling this method, most other methods will be
         *          invalid until accept() has been called again
         */
        void finish();

        /// calls FCGX_Free to free hidden memory associated with the
        /// request object
        void free();

        /// return parameter from the environment
        const char* getParam( const char* name );

        /// return pointer to environment parameters
        char** envp();

        /// return reference to the input stream
        std::istream& in();

        /// return reference to the output stream
        std::ostream& out();

        /// return reference to the error stream
        std::ostream& err();

};


}





#endif // REQUEST_H_
