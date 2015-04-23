/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of ide.
 *
 *  ide is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  ide is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with cpp-pthreads.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   src/cpp-fcgi.cpp
 *
 *  @date   Jan 6, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */
#include <string>

#include <cppformat/format.h>
#include <glog/logging.h>
#include <meerkat/meerkat.h>
#include <tclap/CmdLine.h>

int main(int argc, char **argv) {
  TCLAP::ValueArg<std::string> port_arg(
      "p", "port", "Port number to listen on ", true, "8080", "string");
  TCLAP::ValueArg<std::string> db_path(
      "d", "db_path", "Path to the compilation database", true, "./", "string");

  try {
    TCLAP::CmdLine cmd("Editor Proof of Concept", ' ', "0.1");
    cmd.add(port_arg);
    cmd.parse(argc, argv);
  } catch (TCLAP::ArgException &e) {
    LOG(FATAL) << "error: " << e.error() << " for arg " << e.argId()
               << std::endl;
    return 1;
  }

  struct mg_server *server = mg_create_server(NULL, NULL);
  mg_set_option(server, "document_root", ".");  // Serve current directory
  mg_set_option(server, "listening_port", port_arg.getValue().c_str());

  for (;;) {
    mg_poll_server(server, 1000);  // Infinite loop, Ctrl-C to stop
  }
  mg_destroy_server(&server);

  return 0;
}
