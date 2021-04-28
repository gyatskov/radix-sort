#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

#include "Parameters.h"

template <typename DataType>
struct Dataset;

/**
 * @TODO: Change to span
 *        User should allocate (host) memory
 */
template <typename DataType>
using HostBuffer = std::vector<DataType>;

/// Host buffers passed to the OpenCL kernel
/// @note This should not be owning containers
///       but pointers or a span.
///       Users of the Radix Sort algorithm should provide
///       their own memory for use.
template<typename DataType>
struct HostBuffers
{
	using Parameters = AlgorithmParameters<DataType>;
    using BufferData = HostBuffer<DataType>;
    using BufferAux  = HostBuffer<uint32_t>;

    HostBuffers();
    ~HostBuffers() = default;

    /// Input values
	BufferData m_hKeys;
    /// histograms on the CPU
	BufferAux m_hHistograms;
	/// sum of the local histograms
	BufferAux m_hGlobsum;
	/// permutation
	BufferAux h_Permut;
    /// Output values
	BufferData m_hResultFromGPU;
};

template <typename _DataType>
struct HostData
{
	using DataType      = _DataType;
	using Parameters    = AlgorithmParameters<DataType>;
    using DataBuffer    = HostBuffer<DataType>;
    using DataBuffers   = HostBuffers<DataType>;

    explicit HostData(std::shared_ptr<Dataset<DataType>> dataset);
    HostData() = delete;
    ~HostData() = default;

	// Buffers for reference results
	DataBuffer m_resultSTLCPU;
	DataBuffer m_resultRadixSortCPU;

    /// memory buffers for readbacks of intermediate data
    /// TODO: Make normal object as soon  HostBuffers uses span
    std::shared_ptr<DataBuffers> mHostBuffers;
};

