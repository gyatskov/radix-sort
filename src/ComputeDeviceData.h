#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include "Parameters.h"

#include <vector>
#include <map>
#include <string>

template <typename _DataType>
struct ComputeDeviceData
{
	using DataType   = _DataType;
	using Parameters = AlgorithmParameters<DataType>;

    ComputeDeviceData(cl::Context Context, size_t buffer_size);
    ~ComputeDeviceData() = default;

    /// OpenCL program and kernels
    cl::Program			     m_Program;
    std::vector<std::string> kernelNames;

    /// Maps kernel names to their low-level handles
    std::map<std::string, cl::Kernel> m_kernelMap;
    std::map<std::string, cl::Buffer> m_dMemoryMap;
};

