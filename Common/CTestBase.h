#ifndef _CASSIGNMENT_BASE_H
#define _CASSIGNMENT_BASE_H

#include "IComputeTask.h"

#include "CommonDefs.h"
#include "CArguments.h"
#include "ComputeState.h"

#include <array>

class CTestBase
{
public:
	CTestBase(Arguments arguments = Arguments());

	virtual ~CTestBase();

	//! Main loop. Only to be overridden if something is rendered.
	virtual bool EnterMainLoop();

	//! To be overridden
	virtual bool DoCompute() = 0;

protected:
	virtual bool InitCLContext();

	virtual void ReleaseCLContext();

	virtual bool RunComputeTask(IComputeTask& Task, const std::array<size_t,3>& LocalWorkSize);

    ComputeState m_computeState;

    Arguments m_arguments;
};

#endif // _CASSIGNMENT_BASE_H
