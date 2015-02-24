#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <csignal>
#include <iostream>
#include <set>

#include <boost/format.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

/// Use the boost::format library like sprintf()
template<typename ... Args>
std::string FormatStr(const std::string& format, Args ... args);

/// In the terminal case for the recursive template, the parameter pack is
/// empty and there is nothing left to push into the boost formatter;
inline void PushFormat(boost::format& formatter) {
}

/// Recursive variadic template, inserts the first element into the boost
/// formatter and then recurses on the remainder of the parameter pack.
template<typename Head, typename ... Tail>
inline void PushFormat(boost::format& formatter, Head head, Tail ... tail) {
  formatter % head;
  PushFormat(formatter, tail...);
}

template<typename ... Args>
inline std::string FormatStr(const std::string& format, Args ... args) {
  boost::format formatter(format);
  PushFormat(formatter, args...);
  return formatter.str();
}

DEFINE_string(path, "/tmp/echo_server",
              "The path to the unix domain socket to connect to.");

int main(int argc, char* argv[0]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  LOG_IF(FATAL, sock_fd == -1) << "Failed to open socket : " << strerror(errno);

  sockaddr_un remote_sock;
  remote_sock.sun_family = AF_UNIX;
  strcpy(remote_sock.sun_path, FLAGS_path.c_str());
  int data_len = strlen(remote_sock.sun_path) + sizeof(remote_sock.sun_family);
  LOG_IF(FATAL, connect(sock_fd, (sockaddr*) &remote_sock, data_len) == -1)
      << "Failed to connect to remote: " << strerror(errno);

  std::string line_in;
  std::string line_echo;
  while (std::cin >> line_in) {
    LOG_IF(FATAL,send(sock_fd, line_in.c_str(), line_in.size(), 0) == -1)
        << "Failed to send data over socket";
    const int buf_bytes = 255;
    line_echo.resize(buf_bytes);
    int bytes_received = recv(sock_fd, &line_echo[0], buf_bytes, 0);
    LOG_IF(FATAL,bytes_received < 0) << "Failed to receive, server closed?: "
                                     << strerror(errno);
    line_echo.resize(bytes_received);
    std::cout << "echo> " << line_echo << "\n";
  }

  close(sock_fd);
  return 0;
}
