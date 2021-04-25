#pragma once

#include <vector>
#include <map>
#include <string>
#include <functional>
#include <cstdint>
#include <algorithm>
#include <memory>

#include "Dataset.h"
#include "Parameters.h"

template <typename DataType>
using HostBuffer = std::vector<DataType>;

template <typename _DataType>
struct HostData
{
	using DataType      = _DataType;
	using Parameters    = AlgorithmParameters<DataType>;
    using TypedDataset  = Dataset<DataType>;
    using DataBuffer    = HostBuffer<DataType>;

    explicit HostData(std::shared_ptr<TypedDataset> selectedDataset);

	// results
	DataBuffer m_resultSTLCPU;
	DataBuffer m_resultRadixSortCPU;

	// data sets
	std::shared_ptr<TypedDataset> m_selectedDataset;

	// collector of data sets
	//std::map<std::string, std::vector<DataType>> m_dataSets;

	std::vector<DataType> m_hKeys;
	std::vector<DataType> m_hCheckKeys; // a copy for check
	std::vector<uint32_t> m_hHistograms; // histograms on the CPU
    /// string to Results
	std::map<std::string, DataBuffer> m_hResultGPUMap;
	// sum of the local histograms
	std::vector<uint32_t> m_hGlobsum;
	// permutation
	std::vector<uint32_t> h_Permut;
};
