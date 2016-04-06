/******************************************************************************
                         .88888.   888888ba  dP     dP 
                        d8'   `88  88    `8b 88     88 
                        88        a88aaaa8P' 88     88 
                        88   YP88  88        88     88 
                        Y8.   .88  88        Y8.   .8P 
                         `88888'   dP        `Y88888P' 
                                                       
                                                       
   a88888b.                                         dP   oo                   
  d8'   `88                                         88                        
  88        .d8888b. 88d8b.d8b. 88d888b. dP    dP d8888P dP 88d888b. .d8888b. 
  88        88'  `88 88'`88'`88 88'  `88 88    88   88   88 88'  `88 88'  `88 
  Y8.   .88 88.  .88 88  88  88 88.  .88 88.  .88   88   88 88    88 88.  .88 
   Y88888P' `88888P' dP  dP  dP 88Y888P' `88888P'   dP   dP dP    dP `8888P88 
                                88                                        .88 
                                dP                                    d8888P  
******************************************************************************/

#pragma once

#include "../Common/IComputeTask.h"

#include <vector>
#include <map>
#include <cstdint>

//! A2/T1: Parallel reduction
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
        static const uint32_t _TOTALBITS = 32;  // number of bits for the integer in the list (max=32)
        static const auto _NUM_BITS_PER_RADIX = 4;  // number of bits in the radix
        // max size of the sorted vector
        // it has to be divisible by  _NUM_ITEMS_PER_GROUP * _NUM_GROUPS
        // (for other sizes, pad the list with big values)
        //#define _N (_ITEMS * _GROUPS * 16)  
        static const auto _NUM_MAX_INPUT_ELEMS = (1 << 20);  // maximal size of the list  
        static const auto VERBOSE = true;
        static const auto TRANSPOSE = false; // transpose the initial vector (faster memory access)
        //#define PERMUT  // store the final permutation
        ////////////////////////////////////////////////////////

        // the following parameters are computed from the previous
        static const auto _RADIX = (1 << _NUM_BITS_PER_RADIX); //  radix  = 2^_NUM_BITS_RADIX
        static const auto _NUM_PASSES = (_TOTALBITS / _NUM_BITS_PER_RADIX); // number of needed passes to sort the list
        static const auto _HISTOSIZE = (_NUM_ITEMS_PER_GROUP * _NUM_GROUPS * _RADIX); // size of the histogram
        // maximal value of integers for the sort to be correct
        static const uint32_t _MAXINT = (1 << (_TOTALBITS - 1));
    };

    // Helper methods
	void AllocateDeviceMemory(cl_context Context);
	void CheckLocalMemory(cl_device_id Device);
    void CheckDivisibility();
    void CopyDataToDevice(cl_command_queue CommandQueue);
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

	// input data
	std::vector<DataType> m_hInput;
	// results
	std::vector<DataType> m_resultCPU;

    cl_mem              m_dInputArray;
	cl_mem				m_dResultArray;
    cl_mem              m_dReadWriteArray;

    //OpenCL program and kernels
    cl_program			m_Program;

    std::map<std::string, cl_kernel> m_kernelMap;
    std::map<std::string, std::vector<DataType>> m_hResultGPUMap;
	std::map<std::string, cl_mem> m_dMemoryMap; // NOTE: not used yet

    uint32_t m_hHistograms[Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP]; // histograms on the cpu
    cl_mem m_dHistograms;                   // histograms on the GPU

    // sum of the local histograms
    uint32_t m_hGlobsum[Parameters::_NUM_HISTOSPLIT];
    cl_mem m_dGlobsum;
    cl_mem m_dTemp;  // in case where the sum is not needed

    // list of keys
    uint32_t nkeys; // actual number of keys
    uint32_t nkeys_rounded; // next multiple of _ITEMS*_GROUPS
	std::vector<DataType> m_hKeys;
    std::vector<DataType> m_hCheckKeys; // a copy for check
    cl_mem m_dInKeys;
    cl_mem m_dOutKeys;

    // permutation
    uint32_t h_Permut[Parameters::_NUM_MAX_INPUT_ELEMS];
    cl_mem m_dInPermut;
    cl_mem m_dOutPermut;

	// timers
	float histo_time, scan_time, reorder_time, sort_time, transpose_time;
};
