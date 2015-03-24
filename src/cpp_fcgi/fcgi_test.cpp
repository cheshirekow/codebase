/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of cpp_fcgi.
 *
 *  cpp_fcgi is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  cpp_fcgi is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with cpp_fcgi.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *
 *  @date   Jan 6, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <unistd.h>
#include <iostream>
#include <cpp_fcgi/cpp_fcgi.h>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [SOCKETPATH]|:[PORTNUMBER] \n";
    return 0;
  }

  fcgi::Socket socket;
  fcgi::Request request;

  // initialize the library (same as FCGX_Init() )
  int result = fcgi::init();
  if (result != 0) {
    std::cerr << "Failed to initialize FCGI" << std::endl;
    return 0;
  }

  // open the socket (same as FCGX_OpenSocket )
  result = socket.open(argv[1], 100);
  if (result < 0) {
    std::cerr << "Failed to open socket: " << argv[1] << ", error: " << result
              << "\n";
    return 0;
  }

  // initialize the request object and associate it with the opened socket
  if (request.init(socket) != 0) {
    socket.close();
    std::cerr << "Failed to initialize the request\n";
    return 0;
  }

  // just a count of requests processed
  int reqNum = 0;

  // loop and receive requests
  while (true) {
    // accept a request
    result = request.accept();

    // check the result
    if (result != 0) {
      std::cerr << "Failed to accept request: " << result << std::endl;
      break;
    }

    // write a simple message
    request.out() << "Content-type: text/html\n" << "\n"
                  << "<title>FastCGI Hello! (C, fcgiapp library)</title>"
                  << "<h1>FastCGI Hello! (C, fcgiapp library)</h1>"
                  << "Request number " << ++reqNum << " running on host <i>"
                  << request.getParam("SERVER_NAME") << "</i> Process ID: "
                  << getpid() << "\n";

    // free request memory and prepare for the next one, note this is
    // not stricly necessary as it will happen automatically on the next
    // accept() call if the request hasn't been finished yet
    request.finish();
  }

  request.free();  //< free hidden memory of the request object
  socket.close();  //< close the socket
}
