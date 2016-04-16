#pragma once

#include <vector>
#include <map>
#include <string>
#include <functional>
#include <cstdint>
#include <random>
#include <algorithm>
#include <numeric>

#include "Parameters.h"

template <typename _DataType>
struct HostData {
	using DataType = _DataType;
	using Parameters = Parameters < DataType > ;

	HostData();

	// results
	std::vector<DataType> m_resultSTLCPU;
	std::vector<DataType> m_resultRadixSortCPU;

	std::vector<DataType> m_distributedRandom;
	std::vector<DataType> m_zeros;
	std::vector<DataType> m_invertedRange;

	std::vector<DataType> m_hKeys;
	std::vector<DataType> m_hCheckKeys; // a copy for check
	std::vector<uint32_t> m_hHistograms; // histograms on the CPU
	std::map<std::string, std::vector<DataType>> m_hResultGPUMap;
	// sum of the local histograms
	std::vector<uint32_t> m_hGlobsum;
	// permutation
	std::vector<uint32_t> h_Permut;
};
