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

#ifndef _CREDUCTION_TASK_H
#define _CREDUCTION_TASK_H

#include "../Common/IComputeTask.h"

#include <vector>
#include <map>
#include <cstdint>

//! A2/T1: Parallel reduction
class CRadixSortTask : public IComputeTask
{
public:
	using DataType = int32_t;

	CRadixSortTask(size_t ArraySize);

	virtual ~CRadixSortTask();

	// IComputeTask

	virtual bool InitResources(cl_device_id Device, cl_context Context);
	
	virtual void ReleaseResources();

	virtual void ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);

	virtual void ComputeCPU();

	virtual bool ValidateResults();

protected:

	void RadixSort(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);
    void Histogram(int pass);
    void ScanHistogram();
    void Reorder(int pass);


    void RadixSortReadWrite(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);
    
	void ExecuteTask(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], const std::string& kernel);
	void TestPerformance(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int task);

	//NOTE: we have two memory address spaces, so we mark pointers with a prefix
	//to avoid confusions: 'h' - host, 'd' - device

	size_t		m_N;

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
	std::map<std::string, std::vector<DataType>> m_resultGPUMap;

    struct Parameters {
        ///////////////////////////////////////////////////////
        // these parameters can be changed
        static const auto _ITEMS = 64; // number of items in a group
        static const auto _GROUPS = 16; // the number of virtual processors is _ITEMS * _GROUPS
        static const auto  _HISTOSPLIT = 512; // number of splits of the histogram
        static const auto _TOTALBITS = 30;  // number of bits for the integer in the list (max=32)
        static const auto _BITS = 5;  // number of bits in the radix
        // max size of the sorted vector
        // it has to be divisible by  _ITEMS * _GROUPS
        // (for other sizes, pad the list with big values)
        //#define _N (_ITEMS * _GROUPS * 16)  
        static const auto _N = (1 << 20);  // maximal size of the list  
        static const auto VERBOSE = true;
        static const auto TRANSPOSE = false; // transpose the initial vector (faster memory access)
        //#define PERMUT  // store the final permutation
        ////////////////////////////////////////////////////////

        // the following parameters are computed from the previous
        static const auto _RADIX = (1 << _BITS); //  radix  = 2^_BITS
        static const auto _PASS = (_TOTALBITS / _BITS); // number of needed passes to sort the list
        static const auto _HISTOSIZE = (_ITEMS * _GROUPS * _RADIX); // size of the histogram
        // maximal value of integers for the sort to be correct
        static const auto _MAXINT = (1 << (_TOTALBITS - 1));
    };
};

#endif // _CREDUCTION_TASK_H
