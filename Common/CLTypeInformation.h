#pragma once

template <typename T>
struct TypeNameString
{};

template<>
struct TypeNameString<int16_t> {
	static const char* const stdint_name;
    static const char* const open_cl_name;
};
const char* const TypeNameString<int16_t>::stdint_name  = "int16_t";
const char* const TypeNameString<int16_t>::open_cl_name = "short";

template<>
struct TypeNameString<uint16_t> {
	static const char* const stdint_name;
    static const char* const open_cl_name;
};
const char* const TypeNameString<uint16_t>::stdint_name  = "uint16_t";
const char* const TypeNameString<uint16_t>::open_cl_name = "unsigned short";

template<>
struct TypeNameString<int32_t> {
	static const char* const stdint_name;
    static const char* const open_cl_name;
};
const char* const TypeNameString<int32_t>::stdint_name  = "int32_t";
const char* const TypeNameString<int32_t>::open_cl_name = "int";

template<>
struct TypeNameString<uint32_t> {
	static const char* const stdint_name;
    static const char* const open_cl_name;
};
const char* const TypeNameString<uint32_t>::stdint_name  = "uint32_t";
const char* const TypeNameString<uint32_t>::open_cl_name = "unsigned int";

template<>
struct TypeNameString<int64_t> {
	static const char* const stdint_name;
    static const char* const open_cl_name;
};
const char* const TypeNameString<int64_t>::stdint_name  = "int64_t";
const char* const TypeNameString<int64_t>::open_cl_name = "long";

template<>
struct TypeNameString<uint64_t> {
	static const char* const stdint_name;
    static const char* const open_cl_name;
};
const char* const TypeNameString<uint64_t>::stdint_name  = "uint64_t";
const char* const TypeNameString<uint64_t>::open_cl_name = "unsigned long";