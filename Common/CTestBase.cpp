#include "CTestBase.h"

#include "CLUtil.h"
#include "CTimer.h"

#include <vector>

using namespace std;

#define PRINT_INFO(title, buffer, bufferSize, maxBufferSize, expr) { expr; buffer[bufferSize] = '\0'; std::cout << title << ": " << buffer << std::endl; }
bool ComputeState::init() {
	//////////////////////////////////////////////////////
	//(Sect 4.3)

	// 1. get all platform IDs

	std::vector<cl_platform_id> platformIds;
	constexpr cl_uint c_MaxPlatforms { 16 };
	platformIds.resize(c_MaxPlatforms);

	cl_uint countPlatforms;
	V_RETURN_FALSE_CL(clGetPlatformIDs(c_MaxPlatforms, &platformIds[0], &countPlatforms), "Failed to get CL platform ID");
	platformIds.resize(countPlatforms);

	// 2. find all available GPU devices
	std::vector<cl_device_id> deviceIds;
	constexpr int maxDevices { 16 };
	deviceIds.resize(maxDevices);
	int countAllDevices = 0;


	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

	for (size_t i = 0; i < platformIds.size(); i++)
	{
		// Getting the available devices.
		cl_uint countDevices;
		clGetDeviceIDs(platformIds[i], deviceType, 1, &deviceIds[countAllDevices], &countDevices);
		countAllDevices += countDevices;
	}
	deviceIds.resize(countAllDevices);

	if (countAllDevices == 0)
	{
		std::cout << "No device of the selected type with OpenCL support was found.";
		return false;
	}
	// Choosing the first available device.
	m_CLDevice = deviceIds[0];
	clGetDeviceInfo(m_CLDevice, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &m_CLPlatform, NULL);

	// Printing platform and device data.
	const int maxBufferSize = 1024;
	char buffer[maxBufferSize];
	size_t bufferSize = 0U;
	std::cout << "OpenCL platform:" << std::endl << std::endl;
	PRINT_INFO("Name", buffer, bufferSize, maxBufferSize, clGetPlatformInfo(m_CLPlatform, CL_PLATFORM_NAME, maxBufferSize, (void*)buffer, &bufferSize));
	PRINT_INFO("Vendor", buffer, bufferSize, maxBufferSize, clGetPlatformInfo(m_CLPlatform, CL_PLATFORM_VENDOR, maxBufferSize, (void*)buffer, &bufferSize));
	PRINT_INFO("Version", buffer, bufferSize, maxBufferSize, clGetPlatformInfo(m_CLPlatform, CL_PLATFORM_VERSION, maxBufferSize, (void*)buffer, &bufferSize));
	PRINT_INFO("Profile", buffer, bufferSize, maxBufferSize, clGetPlatformInfo(m_CLPlatform, CL_PLATFORM_PROFILE, maxBufferSize, (void*)buffer, &bufferSize));
	std::cout << std::endl << "Device:" << std::endl << std::endl;
	PRINT_INFO("Name", buffer, bufferSize, maxBufferSize, clGetDeviceInfo(m_CLDevice, CL_DEVICE_NAME, maxBufferSize, (void*)buffer, &bufferSize));
	PRINT_INFO("Vendor", buffer, bufferSize, maxBufferSize, clGetDeviceInfo(m_CLDevice, CL_DEVICE_VENDOR, maxBufferSize, (void*)buffer, &bufferSize));
	PRINT_INFO("Driver version", buffer, bufferSize, maxBufferSize, clGetDeviceInfo(m_CLDevice, CL_DRIVER_VERSION, maxBufferSize, (void*)buffer, &bufferSize));
	cl_ulong localMemorySize;
	clGetDeviceInfo(m_CLDevice, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemorySize, &bufferSize);
	std::cout << "Local memory size: " << localMemorySize << " Byte" << std::endl;
	std::cout << std::endl << "******************************" << std::endl << std::endl;

	cl_int clError;
	m_CLContext = clCreateContext(0, 1, &m_CLDevice, NULL, NULL, &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create OpenCL context.");

	// Finally, create a command queue. All the asynchronous commands to the device will be issued
	// from the CPU into this queue. This way the host program can continue the execution until some results
	// from that device are needed.

	m_CLCommandQueue = clCreateCommandQueue(m_CLContext, m_CLDevice, 0, &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create the command queue in the context");

	return true;
}
#undef PRINT_INFO

void ComputeState::release()
{
	if (m_CLCommandQueue != nullptr)
	{
		clReleaseCommandQueue(m_CLCommandQueue);
		m_CLCommandQueue = nullptr;
	}

	if (m_CLContext != nullptr)
	{
		clReleaseContext(m_CLContext);
		m_CLContext = nullptr;
	}
}

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
