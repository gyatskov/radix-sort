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
struct RuntimesGPU {
    Statistics timeHisto,
               timeScan,
               timeReorder,
               timePaste,
               timeTotal;
};

/// Runtime statistics of CPU implementation algorithms
struct RuntimesCPU {
    Statistics timeRadix,
               timeSTL;
};

template <typename DataType>
class RadixSortGPU
{
public:
    enum class OperationStatus {
        OK,
        HOST_BUFFERS_FAILED,
        INITIALIZATION_FAILED,
        CALCULATION_FAILED,
        CLEANUP_FAILED,
    };

    OperationStatus setHostBuffers(
        const DataType* input,
        size_t length,
        DataType* output
    );
    OperationStatus initialize(cl_context Context, cl_command_queue CommandQueue);
    OperationStatus calculate();
    OperationStatus cleanup();

private:
    /// Performs histogram calculation
	void Histogram(cl_command_queue CommandQueue, int pass);
    /// Performs histogram scan
	void ScanHistogram(cl_command_queue CommandQueue);
    /// Performs reorder step
	void Reorder(cl_command_queue CommandQueue, int pass);
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

    void ComputeCPU() override;

    /** Tests results validity **/
	bool ValidateResults() override;


protected:
    using Parameters = AlgorithmParameters<DataType>;

	// Helper methods
    static std::string BuildOptions();
	void AllocateDeviceMemory(cl_context Context);
	void CheckLocalMemory(cl_device_id Device);
	void CopyDataToDevice(cl_command_queue CommandQueue);
	void CopyDataFromDevice(cl_command_queue CommandQueue);
	void Resize(uint32_t nn);
    /// Pads GPU data buffers
    /// @param CommandQueue OpenCL Command Queue
	void padGPUData(cl_command_queue CommandQueue);

    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////
    /// Performs radix sort algorithm on previously provided data
    /// @param Context OpenCL Context
    /// @param CommandQueue OpenCL Command Queue
    /// @param LocalWorkSize OpenCL Local work size
	void RadixSort(
        cl_context Context,
        cl_command_queue CommandQueue,
        const std::array<size_t,3>& LocalWorkSize
    );
    /// Performs histogram calculation
	void Histogram(cl_command_queue CommandQueue, int pass);
    /// Performs histogram scan
	void ScanHistogram(cl_command_queue CommandQueue);
    /// Performs reorder step
	void Reorder(cl_command_queue CommandQueue, int pass);
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////

	void ExecuteTask(
        cl_context Context,
        cl_command_queue CommandQueue,
        const std::array<size_t,3>& LocalWorkSize,
        const std::string& kernel);

    // list of keys
    uint32_t mNumberKeys; // actual number of keys
    uint32_t mNumberKeysRounded; // next multiple of _ITEMS*_GROUPS

    HostData<DataType>							 mHostData;
    std::shared_ptr<ComputeDeviceData<DataType>> mDeviceData;

	// Runtime statistics GPU
    RuntimesGPU mRuntimesGPU;

	// Runtime statistics CPU
    RuntimesCPU mRuntimesCPU;

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

