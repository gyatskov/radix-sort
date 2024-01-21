#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include "CLUtil.h"

struct ComputeState
{
    ComputeState() = default;

    bool init();

    void release();

	cl_platform_id		m_CLPlatform;
	cl_device_id		m_CLDevice;
    cl::Context			m_CLContext;
	cl_command_queue	m_CLCommandQueue;
};
