#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include <vector>
#include <map>
#include <string>
#include <memory>

enum class MemoryBuffer {
    InputKeys,
    OutputKeys,
    Histograms,
    Globsum,
    InputPermutations,
    OutputPermutations,
    Temp,
};

struct ComputeDeviceData
{
    ComputeDeviceData(cl::Context Context, size_t buffer_size, std::size_t element_size);

    ~ComputeDeviceData() = default;

    /// OpenCL program and kernels
    cl::Program			     m_Program;
    std::vector<std::string> kernelNames;

    /// Maps kernel names to their low-level handles
    std::map<std::string, cl::Kernel> m_kernelMap;
    std::map<MemoryBuffer, cl::Buffer> m_dMemoryMap;
    
    template <typename DataType>
    static std::shared_ptr<ComputeDeviceData> Create(cl::Context Context, size_t buffer_size) {
        return std::make_shared<ComputeDeviceData>(Context, buffer_size, sizeof(DataType));
    } 
};
