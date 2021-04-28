#pragma once

#include "Parameters.h"

#include "../Common/CLUtil.h"

#include <vector>
#include <map>
#include <string>

template <typename _DataType>
struct ComputeDeviceData
{
	using DataType   = _DataType;
	using Parameters = AlgorithmParameters<DataType>;

    ComputeDeviceData(cl_context Context, size_t buffer_size);
    ~ComputeDeviceData();

    //OpenCL program and kernels
    cl_program			     m_Program;
    std::vector<std::string> kernelNames;

    /// Maps kernel names to their low-level handles
    std::map<std::string, cl_kernel> m_kernelMap;
    std::map<std::string, cl_mem>    m_dMemoryMap;
};

