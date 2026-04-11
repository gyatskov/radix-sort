#pragma once

#include <cstdint>
#include <limits>

namespace AlgorithmConfiguration {
	////////////////////////////////////////////////////////
	// Configurable parameters
	////////////////////////////////////////////////////////
    /// Number of items in a group
	inline static constexpr auto _NUM_ITEMS_PER_GROUP = 64U;
    /// Number of virtual processors is _NUM_ITEMS_PER_GROUP * _NUM_GROUPS
	inline static constexpr auto _NUM_GROUPS = 16U;
    /// Total number of items
	inline static constexpr auto _NUM_ITEMS = _NUM_ITEMS_PER_GROUP * _NUM_GROUPS;
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
	// Configuration-derived parameters
	////////////////////////////////////////////////////////
    /// Radix  = 2^_NUM_BITS_RADIX
	inline static constexpr auto _RADIX = (1U << _NUM_BITS_PER_RADIX);
    /// Size of histogram
	inline static constexpr auto _HISTOSIZE = (_NUM_ITEMS_PER_GROUP * _NUM_GROUPS * _RADIX);
	///
    /// Check divisibility of works to assign correct amounts of work to groups/work-items.
    static_assert(AlgorithmConfiguration::_RADIX == 1 << AlgorithmConfiguration::_NUM_BITS_PER_RADIX);
    static_assert(AlgorithmConfiguration::_NUM_MAX_INPUT_ELEMS % (AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP) == 0);
    static_assert((AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP * AlgorithmConfiguration::_RADIX) % AlgorithmConfiguration::_NUM_HISTOSPLIT == 0);
}

/// Collection of compile-time parameters.
///
/// @tparam _DataType Type of data to be sorted
template <typename _DataType>
struct AlgorithmParameters
{
	////////////////////////////////////////////////////////
	// Datatype dependent parameters
	////////////////////////////////////////////////////////
	using DataType = _DataType;
    /// number of bits for the processed integer  type
	inline static constexpr uint32_t _TOTALBITS = sizeof(DataType) << 3U;
    /// Number of needed passes to sort the list
	inline static constexpr auto _NUM_PASSES = (_TOTALBITS / AlgorithmConfiguration::_NUM_BITS_PER_RADIX);
	/// maximum value of integers for the sort to be correct
	inline static constexpr DataType _MAXINT = std::numeric_limits<DataType>::max();
	////////////////////////////////////////////////////////

    static_assert(_TOTALBITS % AlgorithmConfiguration::_NUM_BITS_PER_RADIX == 0);
};

