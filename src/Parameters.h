#pragma once

#include <cstdint>
#include <limits>

/// Collection of compile-time parameters.
///
/// @tparam _DataType Type of data to be sorted
template <typename _DataType>
struct AlgorithmParameters
{
	using DataType = _DataType;
	////////////////////////////////////////////////////////
	// Configurable parameters
	////////////////////////////////////////////////////////
    /// Number of items in a group
	inline static constexpr auto _NUM_ITEMS_PER_GROUP = 64U;
    /// Number of virtual processors is _NUM_ITEMS_PER_GROUP * _NUM_GROUPS
	inline static constexpr auto _NUM_GROUPS = 16U;
    /// number of splits of the histogram
	inline static constexpr auto _NUM_HISTOSPLIT = 512U;
    /// Number of bits in the radix
	inline static constexpr auto _NUM_BITS_PER_RADIX = 4U;
	/// Max size of the sorted vector
	/// @note Must be divisible by  _NUM_ITEMS_PER_GROUP * _NUM_GROUPS
	/// (for other sizes, pad the vector with inf values)
	inline static constexpr auto _NUM_MAX_INPUT_ELEMS = (1U << 25U);
	////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////
	// Datatype-derived parameters
	////////////////////////////////////////////////////////
    /// number of bits for the processed integer  type
	inline static constexpr uint32_t _TOTALBITS = sizeof(DataType) << 3U;
	/// maximum value of integers for the sort to be correct
	inline static constexpr DataType _MAXINT = std::numeric_limits<DataType>::max();
	////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////
	// Configuration-derived parameters
	////////////////////////////////////////////////////////
    /// Radix  = 2^_NUM_BITS_RADIX
	inline static constexpr auto _RADIX = (1U << _NUM_BITS_PER_RADIX);
    /// Number of needed passes to sort the list
	inline static constexpr auto _NUM_PASSES = (_TOTALBITS / _NUM_BITS_PER_RADIX);
    /// Size of histogram
	inline static constexpr auto _HISTOSIZE = (_NUM_ITEMS_PER_GROUP * _NUM_GROUPS * _RADIX);
    /// Number of iterations for performance testing
    /// @TODO: Make configurable at runtime
	inline static constexpr auto _NUM_PERFORMANCE_ITERATIONS = 5U;
	////////////////////////////////////////////////////////
};

