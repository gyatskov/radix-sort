#pragma once

#include "../Common/IComputeTask.h"
#include "ComputeDeviceData.h"

#include <vector>
#include <map>
#include <cstdint>

/// Parallel radix sort
class CRadixSortTask : public IComputeTask
{
public:
    using DataType = uint32_t;

	CRadixSortTask(size_t ArraySize);

	virtual ~CRadixSortTask();

	// IComputeTask
	virtual bool InitResources(cl_device_id Device, cl_context Context);
	virtual void ReleaseResources();
	virtual void ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);
	virtual void ComputeCPU();
	virtual bool ValidateResults();

protected:
	struct Parameters {
		///////////////////////////////////////////////////////
		// these parameters can be changed
		static const auto _NUM_ITEMS_PER_GROUP = 64; // number of items in a group
		static const auto _NUM_GROUPS = 16; // the number of virtual processors is _NUM_ITEMS_PER_GROUP * _NUM_GROUPS
		static const auto _NUM_HISTOSPLIT = 512; // number of splits of the histogram
		//static const uint32_t _TOTALBITS = 32;  // number of bits for the integer in the list (max=32)
        static const uint32_t _TOTALBITS = sizeof(DataType) << 3;  // number of bits for the integer in
		static const auto _NUM_BITS_PER_RADIX = 4;  // number of bits in the radix
		// max size of the sorted vector
		// it has to be divisible by  _NUM_ITEMS_PER_GROUP * _NUM_GROUPS
		// (for other sizes, pad the list with big values) 
		static const auto _NUM_MAX_INPUT_ELEMS = (1 << 19);  // maximal size of the list  
		static const auto VERBOSE = true;
		static const auto TRANSPOSE = false; // transpose the initial vector (faster memory access)
		//#define PERMUT  // store the final permutation
		////////////////////////////////////////////////////////

		// the following parameters are computed from the previous
		static const auto _RADIX = (1 << _NUM_BITS_PER_RADIX); //  radix  = 2^_NUM_BITS_RADIX
		static const auto _NUM_PASSES = (_TOTALBITS / _NUM_BITS_PER_RADIX); // number of needed passes to sort the list
		static const auto _HISTOSIZE = (_NUM_ITEMS_PER_GROUP * _NUM_GROUPS * _RADIX); // size of the histogram
		// maximal value of integers for the sort to be correct
		static const DataType _MAXINT = (1 << (_TOTALBITS - 1));
        // static const DataType _MAXINT_2 = std::numeric_limits<DataType>::max(); // VS13 does not support constexpr yet ;_;
	};

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

    struct HostData {
        HostData() :
            m_hKeys(Parameters::_NUM_MAX_INPUT_ELEMS),
            m_hCheckKeys(Parameters::_NUM_MAX_INPUT_ELEMS),
            h_Permut(Parameters::_NUM_MAX_INPUT_ELEMS),
            m_hHistograms(Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP),
            m_hGlobsum(Parameters::_NUM_HISTOSPLIT),
            m_resultCPU(Parameters::_NUM_MAX_INPUT_ELEMS)
        {}

        // results
        std::vector<DataType> m_resultCPU;
        std::vector<DataType> m_hKeys;
        std::vector<DataType> m_hCheckKeys; // a copy for check
        std::vector<uint32_t> m_hHistograms; // histograms on the CPU
        std::map<std::string, std::vector<DataType>> m_hResultGPUMap;
        // sum of the local histograms
        std::vector<uint32_t> m_hGlobsum;
        // permutation
        std::vector<uint32_t> h_Permut;
    } hostData;

    ComputeDeviceData deviceData;

	// timers
	float histo_time, scan_time, reorder_time, sort_time, transpose_time;
};
