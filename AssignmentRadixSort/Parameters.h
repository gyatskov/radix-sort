#pragma once

#include <cstdint>

struct Parameters {
	using DataType = uint64_t;

	///////////////////////////////////////////////////////
	// these parameters can be changed
	static const auto _NUM_ITEMS_PER_GROUP = 64; // number of items in a group
	static const auto _NUM_GROUPS = 16; // the number of virtual processors is _NUM_ITEMS_PER_GROUP * _NUM_GROUPS
	static const auto _NUM_HISTOSPLIT = 512; // number of splits of the histogram
	//static const uint32_t _TOTALBITS = 32;  // number of bits for the integer in the list (max=32)
	static const uint32_t _TOTALBITS = sizeof(DataType) << 3;  // number of bits for the integer in
	static const auto _NUM_BITS_PER_RADIX = 4;  // number of bits in the radix
	// max size of the sorted vector
	// it has to be divisible by  _NUM_ITEMS_PER_GROUP * _NUM_GROUPS
	// (for other sizes, pad the list with big values) 
	static const auto _NUM_MAX_INPUT_ELEMS = (1U << 16U);  // maximal size of the list  
	static const auto VERBOSE   = false;
	static const auto TRANSPOSE = false; // transpose the initial vector (faster memory access)
	//#define PERMUT  // store the final permutation
	////////////////////////////////////////////////////////

	// the following parameters are computed from the previous
	static const auto _RADIX = (1 << _NUM_BITS_PER_RADIX); //  radix  = 2^_NUM_BITS_RADIX
	static const auto _NUM_PASSES = (_TOTALBITS / _NUM_BITS_PER_RADIX); // number of needed passes to sort the list
	static const auto _HISTOSIZE = (_NUM_ITEMS_PER_GROUP * _NUM_GROUPS * _RADIX); // size of the histogram
	// maximal value of integers for the sort to be correct
	static const DataType _MAXINT = (1ULL << (_TOTALBITS - 1ULL));
	// static const DataType _MAXINT_2 = std::numeric_limits<DataType>::max(); // VS13 does not support constexpr yet ;_;
};