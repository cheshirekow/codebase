#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include <algorithm>
#include <fstream>
#include <map>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <json/json.h>
#include <cppformat/format.h>

DEFINE_string(config_file_path, "~/.throttle.json",
              "Path to the configuration file to use");

#define MAP_ENTRY(x) \
  { #x, x }

std::map<std::string, int> kSocketDomainMap = {  //
    MAP_ENTRY(AF_UNIX),                          //
    MAP_ENTRY(AF_LOCAL),                         //
    MAP_ENTRY(AF_INET), MAP_ENTRY(AF_INET6)};

template <typename T>
T GetMapEntry(const std::map<std::string, T>& map, const std::string& key) {
  auto iter = map.find(key);
  CHECK(iter != map.end()) << "Failed to find key " << key << " in map ";
  return iter->second;
}

json::JSON GetDefaultConfig() {
  // clang-format off
  json::JSON default_config{
    {"max_connections_to_queue", 10},
    {"max_connections_to_accept", 10},
    {"listen" , {
      {"domain", "AF_INET"},
      {"addr", "0.0.0.0"},
      {"port", 8081},
    }}
  };
  // clang-format on

  return default_config;
}

json::JSON GetConfig() {
  if (FLAGS_config_file_path.size() == 0) {
    LOG(FATAL) << "Empty path to config file";
  }

  std::string config_file_path;
  if (FLAGS_config_file_path[0] == '~') {
    config_file_path = fmt::format("{}/{}", std::getenv("HOME"),
                                   FLAGS_config_file_path.substr(2));

  } else {
    config_file_path = FLAGS_config_file_path;
  }

  std::ifstream config_file(config_file_path);
  return json::JSON::parse(config_file);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  json::JSON config;
  try {
    config = GetConfig();
  } catch (const std::exception& ex) {
    LOG(FATAL) << "Faile to get config file: " << ex.what();
  }

  int max_connections_to_queue = config["max_connections_to_queue"];
  int max_connections_to_accept = config["max_connections_to_accept"];

  int socket_domain = GetMapEntry(kSocketDomainMap, config["listen"]["domain"]);
  int sockfd = socket(socket_domain, SOCK_STREAM | SOCK_NONBLOCK, 0);
  PCHECK(sockfd != -1) << "Failed to open a socket";

  struct sockaddr* sockaddr_impl = nullptr;
  socklen_t sockaddr_len = 0;

  struct sockaddr_in sockaddr_inet {};
  struct sockaddr_in6 sockaddr_inet6 {};
  struct sockaddr_un sockaddr_unix {};
  std::string address_string = config["listen"]["addr"];

  switch (socket_domain) {
    case AF_UNIX: {
      sockaddr_unix.sun_family = AF_UNIX;
      // man7.org says that sun_path contains 108 chars, and strncat will copy
      // n+1 bytes (including null terminating character
      const size_t kSunPathMaxChars = static_cast<size_t>(107);
      size_t bytes_to_copy = std::min(address_string.size(), kSunPathMaxChars);
      strncat(sockaddr_unix.sun_path, address_string.c_str(), bytes_to_copy);
      sockaddr_impl = reinterpret_cast<struct sockaddr*>(&sockaddr_unix);
      sockaddr_len = sizeof(sockaddr_unix);
      break;
    }

    case AF_INET: {
      sockaddr_inet.sin_family = AF_INET;
      sockaddr_inet.sin_port = config["listen"]["port"];
      std::string ip_address_string = config["listen"]["addr"];
      PCHECK(inet_aton(ip_address_string.c_str(), &sockaddr_inet.sin_addr))
          << "Failed to parse ip address: " << ip_address_string;
      sockaddr_impl = reinterpret_cast<struct sockaddr*>(&sockaddr_inet);
      sockaddr_len = sizeof(sockaddr_inet);
      break;
    }

    case AF_INET6: {
      sockaddr_inet6.sin6_family = AF_INET6;
      sockaddr_inet6.sin6_port = config["listen"]["port"];
      std::string ip_address_string = config["listen"]["addr"];
      PCHECK(inet_pton(AF_INET6, ip_address_string.c_str(),
                       &sockaddr_inet6.sin6_addr) != -1)
          << "Failed to parse ip address: " << ip_address_string;
      sockaddr_impl = reinterpret_cast<struct sockaddr*>(&sockaddr_inet6);
      sockaddr_len = sizeof(sockaddr_inet6);
      break;
    }
  }

  PCHECK(bind(sockfd, sockaddr_impl, sockaddr_len) != -1)
      << "Failed to bind socket to address";
  PCHECK(listen(sockfd, max_connections_to_queue) != -1)
      << "Failed to mark socket for listen";

  // use epoll and call accept if inbound connection

  LOG_IF(WARNING, close(sockfd) != 0) << "Failed to close the socket, weird";
}
