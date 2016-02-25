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
};

#endif // _CREDUCTION_TASK_H
