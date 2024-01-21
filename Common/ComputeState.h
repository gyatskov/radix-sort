#include <vector>
#include <CL/opencl.hpp>

/**
 * Should exist only once
 */
struct ComputeState
{
    ComputeState() = default;
    ~ComputeState() = default;

    bool init();

    cl::Platform platform();
    cl::Device device();

    cl::Context			m_CLContext;
    cl::CommandQueue	m_CLCommandQueue;
    std::vector<cl::Platform> m_CLPlatforms;
    std::vector<cl::Device>   m_CLDevices;
};
