#pragma once

#include "../Common/IComputeTask.h"
#include "Parameters.h"
#include "HostData.h"
#include "RadixSortOptions.h"
#include "Statistics.h"

#include <vector>
#include <map>
#include <memory>
#include <cstdint>

template <typename DataType>
struct ComputeDeviceData;

struct RuntimesGPU {
    Statistics timeHisto,
               timeScan,
               timeReorder,
               timePaste,
               timeTotal;
};

struct RuntimesCPU {
    Statistics timeRadix,
               timeSTL;
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

    /** Sorts data on CPU **/
	void ComputeCPU() override;

    /** Tests results validity **/
	bool ValidateResults() override;

protected:
    using Parameters = AlgorithmParameters<DataType>;

	// Helper methods
    static std::string BuildOptions();
	void AllocateDeviceMemory(cl_context Context);
	void CheckLocalMemory(cl_device_id Device);
	void CheckDivisibility();
	void CopyDataToDevice(cl_command_queue CommandQueue);
	void CopyDataFromDevice(cl_command_queue CommandQueue);
	void Resize(uint32_t nn);
	void padGPUData(cl_command_queue CommandQueue);

	void RadixSort(cl_context Context, cl_command_queue CommandQueue, const std::array<size_t,3>& LocalWorkSize);
	void Histogram(cl_command_queue CommandQueue, int pass);
	void ScanHistogram(cl_command_queue CommandQueue, int pass);
	void Reorder(cl_command_queue CommandQueue, int pass);

	void ExecuteTask(
        cl_context Context,
        cl_command_queue CommandQueue,
        const std::array<size_t,3>& LocalWorkSize,
        const std::string& kernel);


    /** Writes performance to stream **/

	//NOTE: we have two memory address spaces, so we mark pointers with a prefix
	//to avoid confusions: 'h' - host, 'd' - device

    // list of keys
    uint32_t mNumberKeys; // actual number of keys
    uint32_t mNumberKeysRounded; // next multiple of _ITEMS*_GROUPS
	uint32_t mNumberKeysRest; // rest to fit to number of gpu processors

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

template <typename Stream>
void writePerformance(
    Stream&& stream,
    const RuntimesGPU& runtimesGPU,
    const RuntimesCPU& runtimesCPU,
    size_t numberKeys,
    const std::string& datasetName,
    const std::string& datatype

);

