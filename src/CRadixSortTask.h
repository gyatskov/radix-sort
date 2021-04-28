#pragma once

#include "../Common/IComputeTask.h"
#include "Parameters.h"
#include "HostData.h"
#include "RadixSortOptions.h"
#include "Statistics.h"

#include <map>
#include <memory>
#include <cstdint>

template <typename DataType>
struct ComputeDeviceData;

/// Runtime statistics of GPU implementation algorithms
/// @note Radix sort specific
struct RuntimesGPU {
    Statistics timeHisto{};
    Statistics timeScan{};
    Statistics timeReorder{};
    Statistics timePaste{};
    Statistics timeTotal{};
};

/// Runtime statistics of CPU implementation algorithms
struct RuntimesCPU {
    Statistics timeRadix{};
    Statistics timeSTL{};
};

/// @note Radix sort specific
enum class OperationStatus {
    OK,
    HOST_BUFFERS_FAILED,
    INITIALIZATION_FAILED,
    CALCULATION_FAILED,
    CLEANUP_FAILED,
    RESIZE_FAILED,
    KERNEL_CREATION_FAILED,
    PROGRAM_CREATION_FAILED,
    LOADING_SOURCE_FAILED,
};

/// TODO: Split into different file
/// @note Radix sort specific
/// TODO: Avoid clFinish calls
///       For profiling use clGetEventProfilingInfo api
/// TODO: Replace clUtil with cl.hpp API
template <typename DataType>
class RadixSortGPU
{
public:
    OperationStatus initialize(
        cl_device_id Device,
        cl_context Context,
        uint32_t nn,
        std::shared_ptr<HostBuffers<DataType>> hostBuffers);
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    /// Performs radix sort algorithm on previously provided data
    /// @param CommandQueue OpenCL Command Queue
	OperationStatus calculate( cl_command_queue CommandQueue);
    OperationStatus cleanup();

    void setLogStream(std::ostream* out) noexcept;

    ///
    /// @return Possibly rounded up number of elements
	uint32_t Resize(uint32_t nn);

    /// Pads GPU data buffers
    /// @param CommandQueue OpenCL Command Queue
    /// @param paddingOffset Padding offset in bytes
	void padGPUData(
        cl_command_queue CommandQueue,
        size_t paddingOffset);

    RuntimesGPU getRuntimes() const;

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
    std::shared_ptr<HostBuffers<DataType>> mHostData;

	// Runtime statistics GPU
    RuntimesGPU mRuntimesGPU{};

    // list of keys
    uint32_t mNumberKeysRounded{0U}; // next multiple of _ITEMS*_GROUPS

    /// log stream used for debugging
    std::ostream* mOutStream{nullptr};
};

/// Parallel radix sort
template <typename _DataType>
class CRadixSortTask : public IComputeTask
{
public:
	using DataType = _DataType;

    CRadixSortTask(
        const RadixSortOptions& options,
        std::shared_ptr<Dataset<DataType>> dataset);

	virtual ~CRadixSortTask();

	// IComputeTask
	bool InitResources(cl_device_id Device, cl_context Context) override;
	void ReleaseResources() override;
	void ComputeGPU(
        cl_context Context,
        cl_command_queue CommandQueue,
        const std::array<size_t,3>& LocalWorkSize
    ) override;

    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    void ComputeCPU() override;

    /** Tests results validity **/
	bool ValidateResults() override;
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////


protected:
    using Parameters = AlgorithmParameters<DataType>;

	// Helper methods
	void CheckLocalMemory(cl_device_id Device);
	uint32_t Resize(uint32_t nn);

    /// Performs reorder step
	void Reorder(cl_command_queue CommandQueue, int pass);
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////

	void ExecuteTask(
        cl_context Context,
        cl_command_queue CommandQueue,
        const std::array<size_t,3>& LocalWorkSize
        );


    uint32_t mNumberKeys{0U}; // actual number of keys
    uint32_t mNumberKeysRounded{0U}; // next multiple of _ITEMS*_GROUPS

    HostData<DataType> mHostData;

	// data set
    using TypedDataset = Dataset<DataType>;
	std::shared_ptr<TypedDataset> m_selectedDataset;

	// Runtime statistics CPU
    RuntimesCPU mRuntimesCPU{};

    /// Main GPU Radix Sort algorithm
    RadixSortGPU<DataType> mRadixSortGPU;
    /// Options provided by user
    RadixSortOptions mOptions;
};

/** Measures task performance **/
template<class Callable>
void TestPerformance(
    cl_command_queue CommandQueue,
    Callable&& fun,
    const RadixSortOptions& options,
    const size_t numIterations,
    size_t numberKeys,
    const std::string& datasetName,
    const std::string& datatype
);

/** Writes performance to stream **/
template <typename Stream>
void writePerformance(
    Stream&& stream,
    const RuntimesGPU& runtimesGPU,
    const RuntimesCPU& runtimesCPU,
    size_t numberKeys,
    const std::string& datasetName,
    const std::string& datatype

);

