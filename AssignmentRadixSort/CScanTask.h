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

#ifndef _CSCAN_TASK_H
#define _CSCAN_TASK_H

#include "../Common/IComputeTask.h"

//! A2 / T2 Parallel prefix sum (scan)
class CScanTask : public IComputeTask
{
public:
	//! The second parameter is necessary to pre-allocate the multi-level arrays
	CScanTask(size_t ArraySize, size_t MinLocalWorkSize);

	virtual ~CScanTask();

	// IComputeTask
	virtual bool InitResources(cl_device_id Device, cl_context Context);
	
	virtual void ReleaseResources();

	virtual void ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);

	virtual void ComputeCPU();

	virtual bool ValidateResults();

protected:

	void Scan_Naive(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);
	void Scan_WorkEfficient(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);

	void ValidateTask(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int Task);
	void TestPerformance(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int Task);

	unsigned int		m_N;

	//float data on the CPU
	unsigned int		*m_hArray;

	unsigned int		*m_hResultCPU;
	unsigned int		*m_hResultGPU;
	bool				m_bValidationResults[2];

	// ping-pong arrays for the naive scan
	cl_mem				m_dPingArray;
	cl_mem				m_dPongArray;

	// arrays for each level of the work-efficient scan
	size_t				m_MinLocalWorkSize;
	unsigned int		m_nLevels;
	cl_mem				*m_dLevelArrays;

	//OpenCL program and kernels
	cl_program			m_Program;
	cl_kernel			m_ScanNaiveKernel;
	cl_kernel			m_ScanWorkEfficientKernel;
	cl_kernel			m_ScanWorkEfficientAddKernel;
};

#endif // _CSCAN_TASK_H
