#pragma once

#include "../Common/IComputeTask.h"
#include "ComputeDeviceData.h"
#include "HostData.h"
#include "Parameters.h"

#include <vector>
#include <map>
#include <cstdint>

/// Parallel radix sort
class CRadixSortTask : public IComputeTask
{
public:
	using DataType = Parameters::DataType;

	CRadixSortTask(size_t ArraySize);

	virtual ~CRadixSortTask();

	// IComputeTask
	virtual bool InitResources(cl_device_id Device, cl_context Context);
	virtual void ReleaseResources();
	virtual void ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);
	virtual void ComputeCPU();
	virtual bool ValidateResults();

protected:
	// Helper methods
    std::string buildOptions();
	void AllocateDeviceMemory(cl_context Context);
	void CheckLocalMemory(cl_device_id Device);
	void CheckDivisibility();
	void CopyDataToDevice(cl_command_queue CommandQueue);
	void CopyDataFromDevice(cl_command_queue CommandQueue);
	void Resize(cl_command_queue CommandQueue, int nn);

	void RadixSort(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);
	void Histogram(cl_command_queue CommandQueue, int pass);
	void ScanHistogram(cl_command_queue CommandQueue);
	void Reorder(cl_command_queue CommandQueue, int pass);
	void Transpose(int nbrow, int nbcol);

	void RadixSortReadWrite(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);

	void ExecuteTask(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], const std::string& kernel);
	void TestPerformance(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int task);

	//NOTE: we have two memory address spaces, so we mark pointers with a prefix
	//to avoid confusions: 'h' - host, 'd' - device

    // list of keys
    uint32_t nkeys; // actual number of keys
    uint32_t nkeys_rounded; // next multiple of _ITEMS*_GROUPS

    HostData		  hostData;
    ComputeDeviceData deviceData;

	// timers
	float histo_time, scan_time, reorder_time, sort_time, transpose_time;
};
