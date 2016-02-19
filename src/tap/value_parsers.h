#pragma once

#include <cstdint>
#include <string>

#include "kwargs.h"
#include "tap_common.h"

namespace tap {

// clang-format off
template <uint8_t SIZE> struct GetUnsignedType;
template <> struct GetUnsignedType<1>{ typedef uint8_t Type; };
template <> struct GetUnsignedType<2>{ typedef uint16_t Type; };
template <> struct GetUnsignedType<4>{ typedef uint32_t Type; };
template <> struct GetUnsignedType<8>{ typedef uint64_t Type; };

  template <uint8_t SIZE> struct GetSignedType;
  template <> struct GetSignedType<1>{ typedef int8_t Type; };
  template <> struct GetSignedType<2>{ typedef int16_t Type; };
  template <> struct GetSignedType<4>{ typedef int32_t Type; };
  template <> struct GetSignedType<8>{ typedef int64_t Type; };
// clang-format on

int ParseValue(char* str, std::string* outval);
int ParseValue(char* str, double* outval);
int ParseValue(char* str, float* outval);
int ParseValue(char* str, uint8_t* outval);
int ParseValue(char* str, uint16_t* outval);
int ParseValue(char* str, uint32_t* outval);
int ParseValue(char* str, uint64_t* outval);
int ParseValue(char* str, int8_t* outval);
int ParseValue(char* str, int16_t* outval);
int ParseValue(char* str, int32_t* outval);
int ParseValue(char* str, int64_t* outval);
int ParseValue(char* str, bool* outval);
int ParseValue(char* str, Nil* outval);

}  // namespace tap
