#include "CLTypeInformation.h

template <typename T>
struct TypeNameString
{};

template<>
struct TypeNameString<int16_t> {
    static const char* const name;
};
const char* const TypeNameString<int16_t>::name = "short";

template<>
struct TypeNameString<uint16_t> {
    static const char* const name;
};
const char* const name = "unsigned short";

template<>
struct TypeNameString<int32_t> {
    static const char* const name;
};
const char* const TypeNameString<int32_t>::name = "int";

template<>
struct TypeNameString<uint32_t> {
    static const char* const name;
};
const char* const TypeNameString<uint32_t>::name = "unsigned int";

template<>
struct TypeNameString<int64_t> {
    static const char* const name;
};
const char* const TypeNameString<int64_t>::name = "long";

template<>
struct TypeNameString<uint64_t> {
    static const char* const name;
};
const char* const TypeNameString<uint64_t>::name = "unsigned long";