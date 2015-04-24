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
#include <fstream>
#include <string>

#include <cppformat/format.h>
#include <glog/logging.h>
#include <json_spirit/json_spirit.h>
#include <meerkat/meerkat.h>
#include <tclap/CmdLine.h>

int main(int argc, char **argv) {
  TCLAP::ValueArg<std::string> port("p", "port", "Port number to listen on ",
                                    true, "8080", "string");
  TCLAP::ValueArg<std::string> db_path("d", "db_path",
                                       "Path to the compilation database", true,
                                       "./compilation_commands.json", "string");
  TCLAP::ValueArg<std::string> file("f", "file", "The file to edit", true,
                                    "foo.cc", "string");

  try {
    TCLAP::CmdLine cmd("Editor Proof of Concept", ' ', "0.1");
    cmd.add(port);
    cmd.add(db_path);
    cmd.parse(argc, argv);
  } catch (TCLAP::ArgException &e) {
    LOG(FATAL) << "error: " << e.error() << " for arg " << e.argId()
               << std::endl;
    return 1;
  }

  std::ifstream db_stream(db_path.getValue());
  if (!db_stream.good()) {
    LOG(FATAL) << "Failed to open compilation database" << db_path.getValue();
    return 1;
  }

  json_spirit::mValue db_root;
  json_spirit::read(db_stream, db_root);
  if (!db_root.type() == json_spirit::array_type) {
    LOG(FATAL) << "Compilation database root is not an array";
    return 1;
  }

  for (auto &value : db_root.get_array()) {
    if (value.type() == json_spirit::obj_type) {
      for (std::string key : {"directory", "command", "file"}) {
        fmt::print("{} : {}\n", key, value.get_obj()[key].get_str());
      }
    } else {
      LOG(WARNING) << "Skipping non object db entry";
    }
  }

  struct mg_server *server = mg_create_server(NULL, NULL);
  mg_set_option(server, "document_root", ".");
  mg_set_option(server, "listening_port", port.getValue().c_str());

  for (;;) {
    mg_poll_server(server, 1000);
  }
  mg_destroy_server(&server);

  return 0;
}
