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
#include <set>

#include <boost/format.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

/// Use the boost::format library like sprintf()
template<typename ... Args>
std::string FormatStr(const std::string& format, Args ... args);

/// In the terminal case for the recursive template, the parameter pack is
/// empty and there is nothing left to push into the boost formatter;
inline void PushFormat(boost::format& formatter) {}

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
    "The path to the unix domain socket to open.");

// Our signal handler will write the signal number to this pipe so we can
// multiplex signal handling.
int g_signal_write_fd = 0;

// Number of times the signal handler has been called
int g_num_signals = 0;

void SignalHandler(int signum) {
  const int max_signals = 5;
  if(++g_num_signals > max_signals) {
    LOG(WARNING) << "Received " << max_signals << " signals, forcing exit";
    exit(1);
  }

  char sig_id = static_cast<char>(signum);
  LOG_IF(WARNING,write(g_signal_write_fd,&sig_id,1) != 1)
      << "Failed to write signal to pipe";
}

struct FdSet :
    public fd_set,
    public std::set<int> {

  int Reset() {
    FD_ZERO(this);
    int n_fds = 0;
    for (int fd : *this) {
      FD_SET(fd, this);
      n_fds = std::max(n_fds,fd);
    }
    return n_fds;
  }

  bool IsSet(int fd) {
    return FD_ISSET(fd, this);
  }
};

int main(int argc, char* argv[0]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  LOG_IF(FATAL, listen_fd == -1)
    << "Failed to open socket for incoming connections: " << strerror(errno);
  int flags = fcntl(listen_fd, F_GETFL, 0);
  LOG_IF(FATAL, flags < 0)
    << "Failed to read flags on the listening fd" << strerror(errno);
  flags |= O_NONBLOCK;
  LOG_IF(FATAL, fcntl(listen_fd, F_SETFL, flags) != 0)
    << "Failed to set listening fd to non-blocking" << strerror(errno);

  sockaddr_un local_sock;
  local_sock.sun_family = AF_UNIX;
  strcpy(local_sock.sun_path, FLAGS_path.c_str());
  unlink(local_sock.sun_path);
  int data_len = strlen(local_sock.sun_path) + sizeof(local_sock.sun_family);
  LOG_IF(FATAL, bind(listen_fd, (sockaddr*) &local_sock, data_len) == -1)
                                                   << "Failed to bind socket: "
                                                   << strerror(errno);
  LOG_IF(FATAL, listen(listen_fd, 5) == -1) << "Failed to listen on socket fd: "
                                               << strerror(errno);
  int pipe_ends[2];
  LOG_IF(FATAL, pipe(pipe_ends) == -1) << "Failed to create a pipe for signals: "
                                       << strerror(errno);
  g_signal_write_fd = pipe_ends[1];
  int signal_read_fd = pipe_ends[0];
  signal(SIGINT, &SignalHandler);
  signal(SIGPIPE, SIG_IGN);
  bool should_quit = false;

  FdSet read_fds;
  FdSet except_fds;
  read_fds.insert(signal_read_fd);
  read_fds.insert(listen_fd);

  while (!should_quit) {
    int n_fds = std::max(read_fds.Reset(),except_fds.Reset()) + 1;
    timeval timeout { 1, 0 };
    int n_ready = select(n_fds, &read_fds, nullptr, &except_fds, &timeout);
    if(n_ready == 0) {
      LOG(INFO) << "Timeout";
      continue;
    }

    // Close the file descriptor for any excepted sockets
    for (int client_fd : except_fds) {
      if (except_fds.IsSet(client_fd)) {
        LOG(WARNING)<< "Exception for client fd: " << client_fd;
        close(client_fd);
        read_fds.erase(client_fd);
        except_fds.erase(client_fd);
      } else if(read_fds.IsSet(client_fd)) {
        char buf[255];
        LOG(INFO) << "Receiving data";
        int bytes_read = recv(client_fd, buf, 255, MSG_DONTWAIT);
        if(bytes_read < 0) {
          LOG(WARNING)
          << "Error during read from client "
          << client_fd << " : " << strerror(errno);
        } else {
          LOG(INFO) << "Writing data";
          int bytes_written = send(client_fd, buf, bytes_read, MSG_DONTWAIT);
          LOG_IF(WARNING, bytes_written < bytes_read) << "Read " << bytes_read
          << " bytes but only wrote back " << bytes_written;
        }
      }
    }

    if (read_fds.IsSet(listen_fd)) {
      LOG(INFO)<< "Incoming client connection";
      int client_fd = accept4(listen_fd, nullptr, 0, SOCK_NONBLOCK);
      if(client_fd == -1) {
        LOG(WARNING) << "Failed to accept client connection: "
        << strerror(errno);
      } else {
        read_fds.insert(client_fd);
        except_fds.insert(client_fd);
      }
    }

    if (read_fds.IsSet(signal_read_fd)) {
      LOG(INFO)<< "Process was signalled, will quit";
      should_quit = true;
    }
  }

  for (int fd : pipe_ends) {
    close(fd);
  }
  close(listen_fd);
  for (int fd : except_fds) {
    close(fd);
  }

  unlink(FLAGS_path.c_str());
  return 0;
}
