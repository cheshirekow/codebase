#ifndef TAB_WM_UTIL_H_
#define TAB_WM_UTIL_H_

#include <X11/Xlib.h>
#include <ostream>
#include <string>
#include <boost/format.hpp>

/// Returns a string describing an X event for debugging purposes.
extern std::string XEventToString(const XEvent& e);

/// Returns the name of an X request code.
extern std::string XRequestCodeToString(unsigned char request_code);

/// Use the boost::format library like sprintf()
template<typename ... Args>
std::string FormatStr(const std::string& format, Args ... args);

/******************************************************************************
 *                        Implementation Details
 *****************************************************************************/

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

#endif // TAB_WM_UTIL_H_
