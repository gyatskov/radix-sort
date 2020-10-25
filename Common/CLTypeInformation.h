#pragma once

#include <cstdint>

template <typename T>
struct TypeNameString
{};

template<>
struct TypeNameString<int16_t> {
	inline static constexpr const char* const stdint_name  = "int16_t";
    inline static constexpr const char* const open_cl_name = "short";
};

template<>
struct TypeNameString<uint16_t> {
	inline static constexpr const char* const stdint_name= "uint16_t";
    inline static constexpr const char* const open_cl_name= "unsigned short";
};

template<>
struct TypeNameString<int32_t> {
	inline static constexpr const char* const stdint_name= "int32_t";
    inline static constexpr const char* const open_cl_name= "int";
};

template<>
struct TypeNameString<uint32_t> {
	inline static constexpr const char* const stdint_name= "uint32_t";
    inline static constexpr const char* const open_cl_name= "unsigned int";
};

template<>
struct TypeNameString<int64_t> {
	inline static constexpr const char* const stdint_name= "int64_t";
    inline static constexpr const char* const open_cl_name= "long";
};

template<>
struct TypeNameString<uint64_t> {
	inline static constexpr const char* const stdint_name= "uint64_t";
    inline static constexpr const char* const open_cl_name= "unsigned long";
};
