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
      *error_code = -1;
      return;
    }
  }
}

int ParseValue(char* str, std::string* outval) {
  *outval = str;
  return 0;
}

static const std::set<char> kFloatCharset = GetCharSet("-0123456789.");

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
  *outval = static_cast<float>(temp);
  if (temp < -std::numeric_limits<float>::max() ||
      temp > std::numeric_limits<float>::max()) {
    return 1;
  }
  return 0;
}

static const std::set<char> kUnsignedCharset = GetCharSet("0123456789");

template <uint8_t SIZE>
struct PowersOfTen {
  typedef typename GetUnsignedType<SIZE>::Type UnsignedType;
  static const UnsignedType values[];
  static const int num_values;
};

// 8-bits
template <>
const int PowersOfTen<1>::num_values = 3;

template <>
const uint8_t PowersOfTen<1>::values[] = {
    1,    //
    10,   //
    100,  //
};

template <>
const int PowersOfTen<2>::num_values = 5;

template <>
const uint16_t PowersOfTen<2>::values[] = {
    1,      //
    10,     //
    100,    //
    1000,   //
    10000,  //
};

// 32-bits
template <>
const int PowersOfTen<4>::num_values = 10;

template <>
const uint32_t PowersOfTen<4>::values[] = {
    1,           //
    10,          //
    100,         //
    1000,        //
    10000,       //
    100000,      //
    1000000,     //
    10000000,    //
    100000000,   //
    1000000000,  //
};

// 64-bits
template <>
const int PowersOfTen<8>::num_values = 20;

template <>
const uint64_t PowersOfTen<8>::values[] = {
    1,                       //
    10,                      //
    100,                     //
    1000,                    //
    10000,                   //
    100000,                  //
    1000000,                 //
    10000000,                //
    100000000,               //
    1000000000,              //
    10000000000,             //
    100000000000,            //
    1000000000000,           //
    10000000000000,          //
    100000000000000,         //
    1000000000000000,        //
    10000000000000000,       //
    100000000000000000,      //
    1000000000000000000,     //
    10000000000000000000ULL  //
};

// NOTE: assumes c already validated to be in the range ['0', '9']
int8_t CharToInt(char c) {
  return c - '0';
}

template <typename Unsigned>
int ParseUnsigned(char* str, Unsigned* outval) {
  typedef PowersOfTen<sizeof(Unsigned)> PowTen;
  const Unsigned kMax = std::numeric_limits<Unsigned>::max();

  int error_code = 0;
  AssertCharSet(str, kUnsignedCharset, &error_code);
  if (error_code) {
    return error_code;
  }

  // chomp zeros
  while (*str == '0') {
    ++str;
  }

  // find the back
  char* end = str;
  while (*end != '\0') {
    ++end;
  }

  // string is empty
  if (end == str) {
    *outval = 0;
    return 0;
  }

  // string contains too many digits for this sized type
  if (end - str > PowTen::num_values) {
    return -1;
  }

  // move the back ptr to the last char
  --end;

  // start at the back and build up the result, we've already asserted that
  // pow_ten
  // wont overflow
  *outval = 0;
  for (int pow_ten = 0; end >= str && pow_ten < PowTen::num_values;
       ++pow_ten, --end) {
    Unsigned multiplier = static_cast<Unsigned>(CharToInt(*end));

    // verify that if we multiply this value by the current power of ten then we
    // will not overflow the desired type
    if ((pow_ten == PowTen::num_values - 1) &&
        ((kMax / PowTen::values[pow_ten]) < multiplier)) {
      return -1;
    }

    Unsigned increment = multiplier * PowTen::values[pow_ten];

    // verify that if we add the increment we will not overflow
    if (kMax - *outval < increment) {
      return -1;
    }

    // add the increment
    *outval += increment;
  }
  return 0;
}

int ParseValue(char* str, uint8_t* outval) {
  return ParseUnsigned(str, outval);
}

int ParseValue(char* str, uint16_t* outval) {
  return ParseUnsigned(str, outval);
}

int ParseValue(char* str, uint32_t* outval) {
  return ParseUnsigned(str, outval);
}

int ParseValue(char* str, uint64_t* outval) {
  return ParseUnsigned(str, outval);
}

template <typename Signed>
int ParseSigned(char* str, Signed* outval) {
  typedef PowersOfTen<sizeof(unsigned)> PowTen;
  const Signed kMin = std::numeric_limits<Signed>::min();
  const Signed kMax = std::numeric_limits<Signed>::max();

  bool is_negative = (str[0] == '-');
  if (is_negative) {
    ++str;
  }

  int error_code = 0;
  AssertCharSet(str, kUnsignedCharset, &error_code);
  if (error_code) {
    return error_code;
  }

  // chomp zeros
  while (*str == '0') {
    ++str;
  }

  // find the back
  char* end = str;
  while (*end != '\0') {
    ++end;
  }

  // string is empty
  if (end == str) {
    *outval = 0;
    return 0;
  }

  // string contains too many digits for this sized type
  if (end - str > PowTen::num_values) {
    return -1;
  }

  // move the back ptr to the last char
  --end;

  // start at the back and build up the result, we've already asserted that
  // pow_ten
  // wont overflow
  *outval = 0;
  if (is_negative) {
    for (int pow_ten = 0; end >= str && pow_ten < PowTen::num_values;
         ++pow_ten, --end) {
      Signed multiplier = -static_cast<Signed>(CharToInt(*end));

      // verify that if we multiply this value by the current power of ten then
      // we will not overflow the desired type
      if ((pow_ten == PowTen::num_values - 1) &&
          ((kMin / PowTen::values[pow_ten]) > multiplier)) {
        return -1;
      }

      Signed increment = multiplier * PowTen::values[pow_ten];

      // verify that if we add the increment we will not overflow
      if (kMin - *outval > increment) {
        return -1;
      }

      // add the increment
      *outval += increment;
    }
  } else {
    for (int pow_ten = 0; end >= str && pow_ten < PowTen::num_values;
         ++pow_ten, --end) {
      Signed multiplier = static_cast<Signed>(CharToInt(*end));

      // verify that if we multiply this value by the current power of ten then
      // we will not overflow the desired type
      if ((pow_ten == PowTen::num_values - 1) &&
          ((kMax / PowTen::values[pow_ten]) < multiplier)) {
        return -1;
      }

      Signed increment = multiplier * PowTen::values[pow_ten];

      // verify that if we add the increment we will not overflow
      if (kMax - *outval < increment) {
        return -1;
      }

      // add the increment
      *outval += increment;
    }
  }

  // the negative half of the support for signed types has one extra value than
  // the positive
  // half. For simplicity, we simply wont support that last possible value

  return 0;
}

int ParseValue(char* str, int8_t* outval) {
  // int8 is also a char, so check to see if it's a single non-numeric character
  // than assume
  // it's a char value to store.
  if (str[1] == '\0') {
    int error_code = 0;
    AssertCharSet(str, kUnsignedCharset, &error_code);
    if (error_code) {
      *outval = str[0];
      return 0;
    }
  }

  return ParseSigned(str, outval);
}

int ParseValue(char* str, int16_t* outval) {
  return ParseSigned(str, outval);
}

int ParseValue(char* str, int32_t* outval) {
  return ParseSigned(str, outval);
}

int ParseValue(char* str, int64_t* outval) {
  return ParseSigned(str, outval);
}

}  // namespace tap
