#ifndef _CASSIGNMENT_BASE_H
#define _CASSIGNMENT_BASE_H

#include "IComputeTask.h"

#include "CommonDefs.h"
#include "CArguments.h"

class CAssignmentBase
{
public:
	CAssignmentBase(Arguments arguments = Arguments());

	virtual ~CAssignmentBase();

	//! Main loop. You only need to overload this if you do some rendering in your assignment.
	virtual bool EnterMainLoop();

	//! You need to overload this to define a specific behavior for your assignments
	virtual bool DoCompute() = 0;

protected:
	virtual bool InitCLContext();

	virtual void ReleaseCLContext();

	virtual bool RunComputeTask(IComputeTask& Task, size_t LocalWorkSize[3]);

	cl_platform_id		m_CLPlatform;
	cl_device_id		m_CLDevice;
	cl_context			m_CLContext;
	cl_command_queue	m_CLCommandQueue;

    Arguments m_arguments;
};

#endif // _CASSIGNMENT_BASE_H
