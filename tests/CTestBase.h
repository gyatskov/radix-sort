#pragma once

#include "../Common/IComputeTask.h"

#include "../Common/CommonDefs.h"
#include "../Common/CArguments.h"
#include "../Common/ComputeState.h"

#include <array>

class CTestBase
{
public:
	CTestBase(Arguments arguments = Arguments());

	virtual ~CTestBase() = default;

	//! To be overridden
	virtual bool DoCompute() = 0;

	virtual bool InitCLContext();

	virtual bool RunComputeTask(
        IComputeTask& Task,
        const std::array<size_t,3>& LocalWorkSize
    );

protected:
    ComputeState m_computeState;

    Arguments m_arguments;
};

