#include "CTestBase.h"

#include "CLUtil.h"
#include "CTimer.h"

#include <vector>

using namespace std;


///////////////////////////////////////////////////////////////////////////////
// CTestBase

CTestBase::CTestBase(Arguments arguments /*= Arguments()*/)
    : m_arguments(arguments)
{
}

CTestBase::~CTestBase()
{
	ReleaseCLContext();
}

bool CTestBase::EnterMainLoop()
{
	if(!InitCLContext())
		return false;

	bool success = DoCompute();

	ReleaseCLContext();

	return success;
}


bool CTestBase::InitCLContext()
{
    return m_computeState.init();
}

void CTestBase::ReleaseCLContext()
{
    m_computeState.release();
}

bool CTestBase::RunComputeTask(IComputeTask& Task, const std::array<size_t,3>& LocalWorkSize)
{
	if(m_computeState.m_CLContext == nullptr)
	{
		std::cerr<<"Error: RunComputeTask() cannot execute because the OpenCL context is null."<<endl;
	}

	if(!Task.InitResources(m_computeState.m_CLDevice, m_computeState.m_CLContext))
	{
		std::cerr << "Error during resource allocation. Aborting execution." <<endl;
		Task.ReleaseResources();
		return false;
	}

	// Compute the golden result.
	cout << "Computing CPU reference result...";
	Task.ComputeCPU();
	cout << "DONE" << endl;

	// Running the same task on the GPU.
	cout << "Computing GPU result..." << endl;

	// Runing the kernel N times. This make the measurement of the execution time more accurate.
	Task.ComputeGPU(m_computeState.m_CLContext, m_computeState.m_CLCommandQueue, LocalWorkSize);
	cout << "DONE" << endl;

	// Validating results.
	if (Task.ValidateResults())
	{
		cout << "GOLD TEST PASSED!" << endl;
	}
	else
	{
		cout << "INVALID RESULTS!" << endl;
	}

	// Cleaning up.
	Task.ReleaseResources();

	return true;
}

///////////////////////////////////////////////////////////////////////////////
