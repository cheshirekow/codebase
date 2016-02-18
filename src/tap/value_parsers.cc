#include <climits>
#include <limits>
#include <set>
#include <string>

#include "value_parsers.h"

namespace tap {

std::set<char> GetCharSet(const std::string& chars) {
  return std::set<char>(chars.begin(), chars.end());
}

void AssertCharSet(char* str, const std::set<char>& charset, int* error_code) {
  for (char* ptr = str; *ptr != '\0'; ++ptr) {
    if (charset.count(*ptr) < 1) {
      *error_code = 1;
      return;
    }
  }
}

int ParseValue(char* str, std::string* outval) {
  *outval = str;
  return 0;
}

static const std::set<char> kFloatCharset = GetCharSet("+-0123456789.");

int ParseValue(char* str, double* outval) {
  int error_code = 0;
  AssertCharSet(str, kFloatCharset, &error_code);
  if (error_code) {
    return error_code;
  }

  // for now
  *outval = atof(str);
  return 0;
}

int ParseValue(char* str, float* outval) {
  int error_code = 0;
  AssertCharSet(str, kFloatCharset, &error_code);
  if (error_code) {
    return error_code;
  }

  // for now
  double temp = atof(str);
  if (temp < -std::numeric_limits<float>::max() ||
      temp > std::numeric_limits<float>::max()) {
    return 1;
  }
  *outval = static_cast<float>(temp);
  return 0;
}

int ParseValue(char* str, uint8_t* outval) {
  return 0;
}

int ParseValue(char* str, uint16_t* outval) {
  return 0;
}

int ParseValue(char* str, uint32_t* outval) {
  return 0;
}

int ParseValue(char* str, uint64_t* outval) {
  return 0;
}

int ParseValue(char* str, int8_t* outval) {
  return 0;
}

int ParseValue(char* str, int16_t* outval) {
  return 0;
}

int ParseValue(char* str, int32_t* outval) {
  return 0;
}

int ParseValue(char* str, int64_t* outval) {
  return 0;
}

}  // namespace tap
