#include "ComputeDeviceData.h"
#include "../Common/CLUtil.h"

#include <cstdint>
#include <iostream>

template <typename DataType>
ComputeDeviceData<DataType>::ComputeDeviceData(cl_context Context, size_t buffer_size) :
    m_Program(NULL) 
{
    kernelNames.emplace_back("histogram");
    kernelNames.emplace_back("scanhistograms");
    kernelNames.emplace_back("pastehistograms");
    kernelNames.emplace_back("reorder");

    alternatives.emplace_back("RadixSort_01");

	// allocate device resources
	cl_int clError;
	TODO("Consider using CL_MEM_HOST_whatever");
	m_dInKeys = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(DataType) * buffer_size, NULL, &clError);
	V_RETURN_CL(clError, "Error allocating device array");
	m_dOutKeys = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(DataType) * buffer_size, NULL, &clError);
	V_RETURN_CL(clError, "Error allocating device array");
	m_dInPermut = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(uint32_t) * buffer_size, NULL, &clError);
	V_RETURN_CL(clError, "Error allocating device array");
	m_dOutPermut = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(uint32_t) * buffer_size, NULL, &clError);
	V_RETURN_CL(clError, "Error allocating device array");
	// allocate the histogram on the GPU
	m_dHistograms = clCreateBuffer(Context, CL_MEM_READ_WRITE,
		sizeof(uint32_t) * Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP, NULL, &clError);
	V_RETURN_CL(clError, "Error allocating device array");

	// allocate the auxiliary histogram on GPU
	m_dGlobsum = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(uint32_t) * Parameters::_NUM_HISTOSPLIT, NULL, &clError);
	V_RETURN_CL(clError, "Error allocating device array");

	// temporary vector when the sum is not needed
	m_dTemp = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(uint32_t) * Parameters::_NUM_HISTOSPLIT, NULL, &clError);
	V_RETURN_CL(clError, "Error allocating device array");
}

template <typename DataType>
ComputeDeviceData<DataType>::~ComputeDeviceData() 
{
    SAFE_RELEASE_MEMOBJECT(m_dInKeys);
    SAFE_RELEASE_MEMOBJECT(m_dOutKeys);
    SAFE_RELEASE_MEMOBJECT(m_dInPermut);
    SAFE_RELEASE_MEMOBJECT(m_dOutPermut);

    for (auto& kernel : m_kernelMap) {
        SAFE_RELEASE_KERNEL(kernel.second);
    }

    SAFE_RELEASE_PROGRAM(m_Program);
}

// Specialize ComputeDeviceData for exactly these four types.
template struct ComputeDeviceData < int32_t >;
template struct ComputeDeviceData < int64_t >;
template struct ComputeDeviceData < uint32_t >;
template struct ComputeDeviceData < uint64_t >;
