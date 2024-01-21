#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <vector>
// TODO: Get rid of the following:
#include "CLUtil.h"

struct ComputeState
{
    ComputeState() = default;
    ~ComputeState() = default;

    bool init();

    cl::Platform platform();
    cl::Device device();

    std::vector<cl::Platform> m_CLPlatforms;
    std::vector<cl::Device>   m_CLDevices;
    cl::Context			m_CLContext;
    cl::CommandQueue	m_CLCommandQueue;
};
