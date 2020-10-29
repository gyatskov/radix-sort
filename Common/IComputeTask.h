#ifndef _ICOMPUTE_TASK_H
#define _ICOMPUTE_TASK_H

#include "CommonDefs.h"

// All OpenCL headers
#if defined(WIN32)
    #include <CL/opencl.h>
#elif defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

class IComputeTask
{
public:
	virtual ~IComputeTask() {};

	//! Init any resources specific to the current task
	virtual bool InitResources(cl_device_id Device, cl_context Context) = 0;

	//! Release everything allocated in InitResources()
	virtual void ReleaseResources() = 0;

	//! Perform calculations on the GPU
	virtual void ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]) = 0;

	//! Compute the "golden" solution on the CPU. The GPU results must be equal to this reference
	virtual void ComputeCPU() = 0;

	//! Compare the GPU solution to the "golden" solution
	virtual bool ValidateResults() = 0;
};

#endif // _ICOMPUTE_TASK_H
