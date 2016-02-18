#pragma once

#include <cstdint>
#include <string>


namespace tap {

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

} // namespace tap
