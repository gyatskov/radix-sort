#include "CTestBase.h"

#include "../Common/CLUtil.h"
#include "../Common/CTimer.h"
#include <array>

///////////////////////////////////////////////////////////////////////////////
// CTestBase

CTestBase::CTestBase(Arguments arguments /*= Arguments()*/)
    : m_arguments(arguments)
{
}

CTestBase::~CTestBase()
{
}

bool CTestBase::InitCLContext()
{
    return m_computeState.init();
}

bool CTestBase::RunComputeTask(IComputeTask& Task, const std::array<size_t,3>& LocalWorkSize)
{
	if(m_computeState.m_CLContext() == nullptr)
	{
		std::cerr<<"Error: RunComputeTask() cannot execute because the OpenCL context is null.\n";
	}

	if(!Task.InitResources(
        m_computeState.device()(),
        m_computeState.m_CLContext())
    )
	{
		std::cerr << "Error during resource allocation. Aborting execution." <<std::endl;
		Task.ReleaseResources();
		return false;
	}

	// Compute the golden result.
    std::cout << "Computing CPU reference result...";
	Task.ComputeCPU();
    std::cout << "DONE" << std::endl;

	// Running the same task on the GPU.
    std::cout << "Computing GPU result..." << std::endl;

	// Runing the kernel N times. This make the measurement of the execution time more accurate.
	Task.ComputeGPU(
            m_computeState.m_CLContext(),
            m_computeState.m_CLCommandQueue(),
            LocalWorkSize);
    std::cout << "DONE" << std::endl;

	// Validating results.
	if (Task.ValidateResults())
	{
        std::cout << "GOLD TEST PASSED!" << std::endl;
	}
	else
	{
        std::cout << "INVALID RESULTS!" << std::endl;
	}

	// Cleaning up.
	Task.ReleaseResources();

	return true;
}

///////////////////////////////////////////////////////////////////////////////
