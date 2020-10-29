#include "CLUtil.h"

/** @TODO: Use cl.hpp */
struct ComputeState
{
    ComputeState() = default;

    bool init();

    void release();

	cl_platform_id		m_CLPlatform;
	cl_device_id		m_CLDevice;
	cl_context			m_CLContext;
	cl_command_queue	m_CLCommandQueue;
};
