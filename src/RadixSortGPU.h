#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include "HostData.h"
#include "Statistics.h"
#include "OperationStatus.h"

#include <memory>
#include <iostream>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

/// Runtime statistics of GPU implementation algorithms
/// @note Radix sort specific
struct RuntimesGPU {
    Statistics timeHisto{};
    Statistics timeScan{};
    Statistics timeReorder{};
    Statistics timePaste{};
    Statistics timeTotal{};
};

template <typename DataType>
struct ComputeDeviceData;

/// TODO: Avoid clFinish calls
///       For profiling use clGetEventProfilingInfo api
///       Provide more granular API:
///        +writeData()
///        +sort()
///        +readData()
template <typename DataType>
class RadixSortGPU
{
public:
    /// 1. Creates program and kernel
    /// 2. Initializes host and device memory
    OperationStatus initialize(
        cl::Device Device,
        cl::Context Context,
        uint32_t nn,
        const HostSpans<DataType>& hostSpans
    );

    /// Copies host data to device
    /// @param CommandQueue OpenCL Command Queue
	OperationStatus uploadData(
        cl::CommandQueue CommandQueue
    );

    /// Performs radix sort algorithm on previously provided data
    /// @param CommandQueue OpenCL Command Queue
	OperationStatus calculate(
        cl::CommandQueue CommandQueue
    );

    /// Copies device data to host
    /// @param CommandQueue OpenCL Command Queue
	OperationStatus downloadData(
        cl::CommandQueue CommandQueue
    );

    /// Frees device buffers
    OperationStatus release();

    /// Sets output log stream
    /// @param[in,out] out Log text stream
    void setLogStream(std::ostream* out) noexcept;

    /// Rounds argument to next multiple of NumItems.
    /// @return Possibly rounded up number of elements
	static uint32_t Resize(uint32_t nn) noexcept;

    /// Pads GPU data buffers
    /// @param CommandQueue OpenCL Command Queue
    /// @param paddingOffset Padding offset in bytes
	void padGPUData(
        cl::CommandQueue CommandQueue,
        size_t paddingOffset
    );

    /// Returns runtimes of individual algorithm steps
    /// @return runtimes of individual algorithm steps
    RuntimesGPU getRuntimes() const;

    /// One-call convenience method: allocates all internal buffers,
    /// initialises the GPU sorter, uploads, sorts, downloads, and cleans up.
    ///
    /// @param device   OpenCL device
    /// @param context  OpenCL context
    /// @param queue    OpenCL command queue
    /// @param input    Data to sort (read-only)
    /// @param[out] output  Will be resized and filled with the sorted result
    /// @return OperationStatus::OK on success
    OperationStatus sort(
        cl::Device device,
        cl::Context context,
        cl::CommandQueue queue,
        std::span<const DataType> input,
        std::vector<DataType>& output
    );

    /// @name Per-step methods for inspecting intermediate buffers
    /// Call these instead of calculate() to run one step at a time
    /// and download intermediate results between steps.
    /// @{

    /// Performs histogram calculation for a single pass
    void Histogram(cl::CommandQueue CommandQueue, int pass);
    /// Performs histogram scan
    void ScanHistogram(cl::CommandQueue CommandQueue);
    /// Performs reorder step for a single pass
    void Reorder(cl::CommandQueue CommandQueue, int pass);

    /// @}

    /// Downloads only the key buffer from device to host
    /// (writes into m_hResultFromGPU span provided at initialization).
    OperationStatus downloadKeys(cl::CommandQueue CommandQueue);

    /// Downloads auxiliary buffers from device to host:
    /// histograms, globsum, input permutations, and output permutations.
    OperationStatus downloadIntermediate(cl::CommandQueue CommandQueue);

private:
    using Parameters = AlgorithmParameters<DataType>;

    static std::string BuildPreamble();
    /// Compiles build options for OpenCL kernel
    static std::string BuildOptions();

	void CopyDataToDevice(cl::CommandQueue CommandQueue);
	void CopyDataFromDevice(cl::CommandQueue CommandQueue);

    /// Device program, kernels and buffers
    std::shared_ptr<ComputeDeviceData<DataType>> mDeviceData;
    /// Pointers to host memory buffers
    HostSpans<DataType> mHostSpans;

	// Runtime statistics GPU
    RuntimesGPU mRuntimesGPU{};

    // list of keys
    uint32_t mNumberKeysRounded{0U}; // next multiple of _ITEMS*_GROUPS

    /// log stream used for debugging
    std::ostream* mOutStream{nullptr};
};
