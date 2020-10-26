#pragma once

#include "../Common/IComputeTask.h"
#include "Parameters.h"
#include "HostData.h"
#include "RadixSortOptions.h"

#include <vector>
#include <map>
#include <memory>
#include <limits>
#include <cstdint>

template <typename DataType>
struct ComputeDeviceData;

struct Statistics
{
    double min;
    double max;
    double avg;
    double sum;

    std::size_t n;

    Statistics() :
        min(std::numeric_limits<decltype(min)>::infinity()),
        max(-std::numeric_limits<decltype(max)>::infinity()),
        avg(0),
        sum(0),
        n(0)
    {}

    void update(double value) {
        n++;
        sum += value;
        avg = sum / n;
        if (value > max) {
            max = value;
        }
        else if (value < min) {
            min = value;
        }
    }
};

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
	virtual void ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);
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

	void RadixSort(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);
	void Histogram(cl_command_queue CommandQueue, int pass);
	void ScanHistogram(cl_command_queue CommandQueue, int pass);
	void Reorder(cl_command_queue CommandQueue, int pass);

	void ExecuteTask(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], const std::string& kernel);
	void TestPerformance(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int task);

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
