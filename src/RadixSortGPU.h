#pragma once

// @TODO: Replace with only ocl include
#include "../Common/IComputeTask.h"

#include "HostData.h"
#include "Statistics.h"
#include "OperationStatus.h"
#include <memory>

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
/// TODO: Replace clUtil with cl.hpp API
template <typename DataType>
class RadixSortGPU
{
public:
    /// 1. Creates program and kernel
    /// 2. Initializes host and device memory
    OperationStatus initialize(
        cl_device_id Device,
        cl_context Context,
        uint32_t nn,
        const HostSpans<DataType>& hostSpans
    );

    /// Performs radix sort algorithm on previously provided data
    /// @param CommandQueue OpenCL Command Queue
	OperationStatus calculate( cl_command_queue CommandQueue);

    /// Frees device buffers
    OperationStatus cleanup();

    /// Sets output log stream
    /// @param[in,out] out Log text stream
    void setLogStream(std::ostream* out) noexcept;

    /// Rounds argument to next multiple of NumItems.
    /// @return Possibly rounded up number of elements
	uint32_t Resize(uint32_t nn) const noexcept;

    /// Pads GPU data buffers
    /// @param CommandQueue OpenCL Command Queue
    /// @param paddingOffset Padding offset in bytes
	void padGPUData(
        cl_command_queue CommandQueue,
        size_t paddingOffset);

    /// Returns runtimes of individual algorithm steps
    /// @return runtimes of individual algorithm steps
    RuntimesGPU getRuntimes() const;

    /// TODO: Add methods to inspect intermediate buffers
    //        between runs.
    //        E.g. providing each step a public API

private:
    using Parameters = AlgorithmParameters<DataType>;

    static std::string BuildPreamble();
    /// Compiles build options for OpenCL kernel
    static std::string BuildOptions();
    /// Performs histogram calculation
	void Histogram(cl_command_queue CommandQueue, int pass);
    /// Performs histogram scan
	void ScanHistogram(cl_command_queue CommandQueue);
    /// Performs reorder step
	void Reorder(cl_command_queue CommandQueue, int pass);

	void CopyDataToDevice(cl_command_queue CommandQueue);
	void CopyDataFromDevice(cl_command_queue CommandQueue);

    std::shared_ptr<ComputeDeviceData<DataType>> mDeviceData;
    HostSpans<DataType> mHostSpans;

	// Runtime statistics GPU
    RuntimesGPU mRuntimesGPU{};

    // list of keys
    uint32_t mNumberKeysRounded{0U}; // next multiple of _ITEMS*_GROUPS

    /// log stream used for debugging
    std::ostream* mOutStream{nullptr};
};
