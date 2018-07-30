#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstdlib>


template <typename DataType>
class RadixSortCPU {
public:
	using Parameters = Parameters < DataType > ;

	struct uint128_t 
	{
		uint64_t low;
		uint64_t high;
	};

	static const auto NUM_BINS = Parameters::_TOTALBITS / Parameters::_NUM_BITS_PER_RADIX;

	// A function to do counting sort of arr[] according to
	// the digit represented by exp.
	template <typename ElemType>
	static void countSort(std::vector<ElemType>& arr, uint64_t exp)
	{
		using UnsignedElemType = std::make_unsigned<ElemType>::type;

		const auto n = static_cast<int64_t>(arr.size());
		std::vector<ElemType> output(n, 0); // output array
		int64_t i = 0;
		std::vector<size_t> count(NUM_BINS, 0);

		/// If operating on signed integers, the minimum value (which is negative) will be added
		UnsignedElemType offset = -std::numeric_limits<ElemType>::min();

		// Store count of occurrences in count[]
		for (i = 0; i < n; i++) {
			const auto elem_value = static_cast<std::make_unsigned<ElemType>::type>(arr[i] + offset);
			count[(elem_value / exp) % NUM_BINS]++;
		}

		// Change count[i] so that count[i] now contains actual
		// position of this digit in output[]
		for (i = 1; i < NUM_BINS; i++) {
			count[i] += count[i - 1];
		}

		// Build the output array
		for (i = n - 1; i >= 0; i--) {
			const auto elem_value = static_cast<std::make_unsigned<ElemType>::type>(arr[i] + offset);
			output[count[( elem_value / exp) % NUM_BINS] - 1] = arr[i];
			count[(elem_value / exp) % NUM_BINS]--;
		}

		// Copy the output array to arr[], so that arr[] now
		// contains sorted numbers according to current digit
		std::copy(output.begin(), output.end(), arr.begin());
	}

	template <typename ElemType>
	static typename std::make_unsigned<ElemType>::type customAbs(ElemType val) 
	{
		return (val < 0) ? (-val) : (val);
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
		const auto max_elem = *std::max_element(arr.begin(), arr.end());

		// Do counting sort for every digit. Note that instead
		// of passing digit number, exp is passed. exp is 10^i
		// where i is current digit number
		auto numDigits = static_cast<uint64_t>(std::ceil(std::log(customAbs(max_elem)) / std::log(NUM_BINS)));
		// TODO: Adapt for signed integers, otherwise log(max_elem) is (minus) infinite
		if (max_elem == 0) {
			numDigits = 1;
		}
		//for (uint64_t exp = 1ULL; std::abs(m) > exp; exp *= Radix) {
		for (uint64_t exp = 0ULL; exp < numDigits; exp++) {
			countSort(arr, static_cast<uint64_t>(std::pow(NUM_BINS, exp)));
		}
	}
};
											