#ifndef _ICOMPUTE_TASK_H
#define _ICOMPUTE_TASK_H

#include "CommonDefs.h"

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include <array>

class IComputeTask
{
public:
	virtual ~IComputeTask() {};

	//! Init any resources specific to the current task
	virtual bool InitResources(cl::Device Device, cl::Context Context) = 0;

	//! Release everything allocated in InitResources()
	virtual void ReleaseResources() = 0;

	//! Perform calculations on the GPU
	virtual void ComputeGPU(
        cl::Context Context,
        cl::CommandQueue CommandQueue,
        const std::array<size_t,3>& LocalWorkSize
    ) = 0;

	//! Compute the "golden" solution on the CPU. The GPU results must be equal to this reference
	virtual void ComputeCPU() = 0;

	//! Compare the GPU solution to the "golden" solution
	virtual bool ValidateResults() = 0;
};

#endif // _ICOMPUTE_TASK_H
