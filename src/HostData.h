#pragma once

#include "Parameters.h"

#include <vector>
#include <memory>
#include <span>
#include <functional>

#include <cstdint>

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
	/// Output permutations (after reorder, before swap)
	BufferAux h_OutputPermut;
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
    std::span<DataType>,
    std::span<uint32_t>
>;

struct HostSpansProxy {
    std::function<void*()> hKeysData;
    std::function<void*()> hHistogramsData;
    std::function<void*()> hGlobsumData;
    std::function<void*()> hPermutData;
    std::function<void*()> hOutputPermutData;
    std::function<void*()> hResultFromGPUData;
    
    template <typename DataType>
    static HostSpansProxy FromHostSpans(HostSpans<DataType>& hostSpans) {
        return HostSpansProxy {
            [hostSpans]{ return static_cast<DataType*>(hostSpans.m_hKeys.data()); },
            [hostSpans]{ return static_cast<uint32_t*>(hostSpans.m_hHistograms.data()); },
            [hostSpans]{ return static_cast<uint32_t*>(hostSpans.m_hGlobsum.data()); },
            [hostSpans]{ return static_cast<uint32_t*>(hostSpans.h_Permut.data()); },
            [hostSpans]{ return static_cast<uint32_t*>(hostSpans.h_OutputPermut.data()); },
            [hostSpans]{ return static_cast<DataType*>(hostSpans.m_hResultFromGPU.data()); }
        };
    }
};

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

