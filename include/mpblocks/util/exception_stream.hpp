#ifndef MPBLOCKS_UTIL_EXCEPTION_STREAM_HPP_
#define MPBLOCKS_UTIL_EXCEPTION_STREAM_HPP_

#include <exception>
#include <sstream>

namespace mpblocks {
namespace  utility {

/// used to simplify the process of generating an exception message
/**
 *  Derives from stringstream so provides an ostream interface, but throws
 *  an exception with the contents of the string when the object is destroyed
 *
 *  \tparam Exception_t must be an exception type which accepts a
 *                      const char* in it's constructor
 *
 */
template <typename Exception_t>
class ExceptionStream : public std::stringstream {
 public:
  ~ExceptionStream() { throw Exception_t(str().c_str()); }

  std::ostream& operator()() { return *this; }
};

typedef ExceptionStream<std::runtime_error> ex;

} // namespace utility
} // namespace mpblocks

#endif  // MPBLOCKS_UTIL_EXCEPTION_STREAM_HPP_
