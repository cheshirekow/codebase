#include <string>

#include <cppformat/format.h>
#include <glog/logging.h>
#include <meerkat/meerkat.h>
#include <tclap/CmdLine.h>

int main(int argc, char **argv) {
  TCLAP::ValueArg<std::string> port_arg(A
      "p", "port", "Port number to listen on ", true, "8080", "string");
  TCLAP::ValueArg<std::string> db_path(
      "d", "db_path", "Path to the compilation database", true, "./", "string");

  try {
    TCLAP::CmdLine cmd("Editor Proof of Concept", ' ', "0.1");
    cmd.add(port_arg);
    cmd.parse(argc, argv);
    port_string = port_arg.getValue();
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
