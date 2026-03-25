#pragma once

#include "Common/IComputeTask.h"

#include "Common/CommonDefs.h"
#include "Common/ComputeState.h"

#include <array>
#include <vector>
#include <string>

class CTestBase
{
public:
	CTestBase(std::vector<std::string> arguments = {});

	virtual ~CTestBase() = default;

	//! To be overridden
	virtual bool DoCompute() = 0;

	virtual bool InitCLContext();

	virtual bool RunComputeTask(
        IComputeTask& Task,
        const LocalWorkSize& LocalWorkSize
    );

protected:
    ComputeState m_computeState;

    std::vector<std::string> m_arguments;
};

