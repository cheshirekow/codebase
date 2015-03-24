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
 *  @file   include/cpp-fcgi.h
 *
 *  @date   Jan 6, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  main include: init(), shutdown() and includes other headers
 */

#ifndef CPP_FCGI_H_
#define CPP_FCGI_H_

#include <fcgiapp.h>
#include <fcgio.h>
#include <cpp_fcgi/Socket.h>
#include <cpp_fcgi/Request.h>


/// object-oriented wrapper around FCGX
namespace fcgi {

/// calls FCGX_Init
int init();

/// calls FCGX_ShutdownPending
void shutdown();

}


#endif // CPP_FCGI_H_
