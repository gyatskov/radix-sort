#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

#include "Parameters.h"
#include "../Common/CheapSpan.h"

template <typename DataType>
struct Dataset;

/// Host buffers passed to the OpenCL kernel
/// @note Radix sort specific
template<
    typename BufferData,
    typename BufferAux
>
struct HostBuffers
{
    /// Input values
	BufferData m_hKeys;
    /// Internal histograms on the CPU
	BufferAux m_hHistograms;
	/// Internal sum of the local histograms
	BufferAux m_hGlobsum;
	/// Internal permutations
	BufferAux h_Permut;
    /// Output values
	BufferData m_hResultFromGPU;
};

template<typename DataType>
using HostData = HostBuffers<
    std::vector<DataType>,
    std::vector<uint32_t>
>;

template<typename DataType>
using HostSpans = HostBuffers<
    CheapSpan<DataType>,
    CheapSpan<uint32_t>
>;

/// @note Only used for tests
template <typename T>
struct HostDataWithReference
{
	using DataType      = T;
	using Parameters    = AlgorithmParameters<DataType>;
    using ResultBuffer  = std::vector<DataType>;

    explicit HostDataWithReference(std::shared_ptr<Dataset<DataType>> dataset);
    HostDataWithReference() = delete;
    ~HostDataWithReference() = default;

	// Real buffers for reference results
	ResultBuffer m_resultSTLCPU;
	ResultBuffer m_resultRadixSortCPU;

    /// Real buffers for readbacks of intermediate data
    HostData<DataType> mHostBuffers;
};

