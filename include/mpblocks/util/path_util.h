#ifndef MPBLOCKS_UTIL_PATH_UTIL_H_
#define MPBLOCKS_UTIL_PATH_UTIL_H_

#include <string>

namespace  mpblocks {
namespace path_util {

// path to the mpblocks source tree
std::string GetSourcePath();

// path to the mpblocks build tree
std::string GetBuildPath();

// path to the mpblocks installed resource tree
std::string GetResourcePath();

// path to the mpblocks runtime settings tree
std::string GetPreferencesPath();


} // namespace path_util
} // namespace mpblocks

#endif  // MPBLOCKS_UTIL_PATH_UTIL_H_
