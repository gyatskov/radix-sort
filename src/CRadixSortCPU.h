#pragma once

#include "Parameters.h"

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

namespace {

/// Returns unsigned absolute value
/// @tparam ElemType Element type
/// @param val Value
/// @return unsigned absolute of val
template <typename ElemType>
inline constexpr
typename std::make_unsigned<ElemType>::type abs(ElemType val)
{
    return (val < 0) ? (-val) : (val);
}

} // namespace


template <typename DataType>
class RadixSortCPU {
public:
	using Parameters = AlgorithmParameters<DataType>;

    static_assert(Parameters::_TOTALBITS % Parameters::_NUM_BITS_PER_RADIX == 0);
	inline static constexpr auto NUM_BINS = Parameters::_TOTALBITS / Parameters::_NUM_BITS_PER_RADIX;

	// A function to do counting sort of arr[] according to
	// the digit represented by exp.
    /// @param arr Vector to be sorted
    /// @param exp Exponent
    ///
    /// @note Allocates memory
	template <typename ElemType>
	static void countSort(std::vector<ElemType>& arr, uint64_t exp)
	{
		using UnsignedElemType = typename std::make_unsigned_t<ElemType>;

		const auto n = arr.size();
		std::vector<ElemType> output(n, 0); // output array
		size_t i = 0;
		std::vector<size_t> count(NUM_BINS, 0);

		/// Offset to shift signed integers into unsigned region
		constexpr auto offset = std::numeric_limits<ElemType>::min();

		// Store count of occurrences in count[]
		for (i = 0; i < n; i++) {
			const auto elem_value = static_cast<UnsignedElemType>(arr[i] - offset);
			count[(elem_value / exp) % NUM_BINS]++;
		}

		// Change count[i] so that count[i] now contains actual
		// position of this digit in output[]
		for (i = 1; i < NUM_BINS; i++) {
			count[i] += count[i - 1];
		}

		// Build the output array
		for (int64_t i = n-1; i >= 0; i--) {
			const auto elem_value = static_cast<UnsignedElemType>(arr[i] - offset);
            const auto countIdx {(elem_value / exp) % NUM_BINS};
			output[count[countIdx] - 1] = arr[i];
			count[countIdx]--;
		}

		// Copy the output array to arr[], so that arr[] now
		// contains sorted numbers according to current digit
		std::copy(output.begin(), output.end(), arr.begin());
	}


	//
	//░░░░░▄▄▄▄▀▀▀▀▀▀▀▀▄▄▄▄▄▄░░░░░░░
	//░░░░░█░░░░▒▒▒▒▒▒▒▒▒▒▒▒░░▀▀▄░░░░
	//░░░░█░░░▒▒▒▒▒▒░░░░░░░░▒▒▒░░█░░░
	//░░░█░░░░░░▄██▀▄▄░░░░░▄▄▄░░░░█░░
	//░▄▀▒▄▄▄▒░█▀▀▀▀▄▄█░░░██▄▄█░░░░█░
	//█░▒█▒▄░▀▄▄▄▀░░░░░░░░█░░░▒▒▒▒▒░█
	//█░▒█░█▀▄▄░░░░░█▀░░░░▀▄░░▄▀▀▀▄▒█
	//░█░▀▄░█▄░█▀▄▄░▀░▀▀░▄▄▀░░░░█░░█░
	//░░█░░░▀▄▀█▄▄░█▀▀▀▄▄▄▄▀▀█▀██░█░░
	//░░░█░░░░██░░▀█▄▄▄█▄▄█▄████░█░░░
	//░░░░█░░░░▀▀▄░█░░░█░█▀██████░█░░
	//░░░░░▀▄░░░░░▀▀▄▄▄█▄█▄█▄█▄▀░░█░░
	//░░░░░░░▀▄▄░▒▒▒▒░░░░░░░░░░▒░░░█░
	//░░░░░░░░░░▀▀▄▄░▒▒▒▒▒▒▒▒▒▒░░░░█░
	//░░░░░░░░░░░░░░▀▄▄▄▄▄░░░░░░░░█░░
	// The main function to that sorts arr[] of size n using
	// Radix Sort
	template<typename ElemType>
	static void sort(std::vector<ElemType>& arr)
	{
		// Find the maximum number to know number of digits
		// in O(nkeys)
		const auto max_elem { *std::max_element(arr.begin(), arr.end()) };

		// Do counting sort for every digit. Note that instead
		// of passing digit number, exp is passed. exp is 10^i
		// where i is current digit number
		const auto numDigits = max_elem ? static_cast<uint64_t>(std::ceil(std::log(abs(max_elem)) / std::log(NUM_BINS))) : 1;
		//for (uint64_t exp = 1ULL; std::abs(m) > exp; exp *= Radix) {
		for (uint64_t exp = 0ULL; exp < numDigits; exp++) {
			countSort(arr, static_cast<uint64_t>(std::pow(NUM_BINS, exp)));
		}
	}
};

