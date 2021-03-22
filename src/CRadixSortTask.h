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

/// Parallel radix sort
template <typename _DataType>
class CRadixSortTask : public IComputeTask
{
public:
	using DataType = _DataType;

	CRadixSortTask(const RadixSortOptions& options, std::shared_ptr<Dataset<DataType>> dataset);

	virtual ~CRadixSortTask();

	// IComputeTask
	virtual bool InitResources(cl_device_id Device, cl_context Context);
	virtual void ReleaseResources();
	virtual void ComputeGPU(cl_context Context, cl_command_queue CommandQueue, const std::array<size_t,3>& LocalWorkSize);
	virtual void ComputeCPU();
	virtual bool ValidateResults();

protected:
    using Parameters = AlgorithmParameters<DataType>;

	// Helper methods
    std::string buildOptions();
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

	void ExecuteTask(cl_context Context, cl_command_queue CommandQueue, const std::array<size_t,3>& LocalWorkSize, const std::string& kernel);
	void TestPerformance(cl_context Context, cl_command_queue CommandQueue, const std::array<size_t,3>& LocalWorkSize, unsigned int task);

    template <typename Stream>
    void writePerformance(Stream&& stream);

	//NOTE: we have two memory address spaces, so we mark pointers with a prefix
	//to avoid confusions: 'h' - host, 'd' - device

    // list of keys
    uint32_t nkeys; // actual number of keys
    uint32_t nkeys_rounded; // next multiple of _ITEMS*_GROUPS
	uint32_t nkeys_rest; // rest to fit to number of gpu processors

    HostData<DataType>							 hostData;
    std::shared_ptr<ComputeDeviceData<DataType>> deviceData;

	// timers
	Statistics histo_time, scan_time, reorder_time, paste_time, sort_time;
    Statistics cpu_radix_time, cpu_stl_time;

    RadixSortOptions options;
};
