/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CReductionTask.h"

#include "../Common/CLUtil.h"
#include "../Common/CTimer.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CReductionTask

string g_kernelNames[] = {
	"RadixSort",
    "RadixSortReadWrite"
};

CReductionTask::CReductionTask(size_t ArraySize)
	: m_N(ArraySize), 
	m_hInput(ArraySize),
	m_resultCPU(ArraySize),
	m_resultGPU(ArraySize),
	m_dResultArray(),
	m_Program(NULL)
{
}

CReductionTask::~CReductionTask()
{
	ReleaseResources();
}

bool CReductionTask::InitResources(cl_device_id Device, cl_context Context)
{
	//CPU resources

	//fill the array with some values
	for (size_t i = 0; i < m_N; i++) {
		//m_hInput[i] = m_N - i;			// Use this for debugging
		m_hInput[i] = rand() & 15;
	}
	//device resources
	cl_int clError, clError2;
    m_dInputArray  = clCreateBuffer(Context, CL_MEM_READ_ONLY,  sizeof(DataType) * m_N, NULL, &clError2);
    m_dResultArray = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, sizeof(DataType) * m_N, NULL, &clError2);
    m_dReadWriteArray = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(DataType) * m_N, NULL, &clError2);
	clError = clError2;
	V_RETURN_FALSE_CL(clError, "Error allocating device arrays");

	//load and compile kernels
	string programCode;
	size_t programSize = 0;

	//std::string pathToRadixSort("\\\\studhome.ira.uka.de\\s_yatsko\\windows\\folders\\Documents\\Visual Studio 2013\\Projects\\paralgo - radix - sort\\AssignmentRadixSort");

	CLUtil::LoadProgramSourceToMemory("RadixSort.cl", programCode);
	m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
    if (m_Program == nullptr) {
        return false;
    }

	//create kernels
    for (const auto& kernelName : g_kernelNames) {
        m_kernelMap[kernelName] = clCreateKernel(m_Program, kernelName.c_str(), &clError);
        std::string errorMsg("Failed to create kernel: ");
        errorMsg += kernelName;
        V_RETURN_FALSE_CL(clError, errorMsg.c_str());
    }

	return true;
}

void CReductionTask::ReleaseResources()
{
	// device resources
    SAFE_RELEASE_MEMOBJECT(m_dInputArray);
	SAFE_RELEASE_MEMOBJECT(m_dResultArray);
    SAFE_RELEASE_MEMOBJECT(m_dReadWriteArray);

    for (auto& kernel : m_kernelMap) {
        SAFE_RELEASE_KERNEL(kernel.second);
    }

	SAFE_RELEASE_PROGRAM(m_Program);
}

void CReductionTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	ExecuteTask(Context, CommandQueue, LocalWorkSize, 0);
    ExecuteTask(Context, CommandQueue, LocalWorkSize, 1);

	TestPerformance(Context, CommandQueue, LocalWorkSize, 0);
    TestPerformance(Context, CommandQueue, LocalWorkSize, 1);
}

void CReductionTask::ComputeCPU()
{
	CTimer timer;
	timer.Start();

	const unsigned int NUM_ITERATIONS = 10;
    for (unsigned int j = 0; j < NUM_ITERATIONS; j++) {
        m_resultCPU = m_hInput;
        // Use this only for really basic testing of the actually pointless kernel
        for (auto& val : m_resultCPU) {
            val *= val;
        }
        // Correct result:
        //std::sort(m_resultCPU.begin(), m_resultCPU.end());
    }
	timer.Stop();
	
    double ms = timer.GetElapsedMilliseconds() / double(NUM_ITERATIONS);
	cout << "  average time: " << ms << " ms, throughput: " << 1.0e-6 * (double)m_N / ms << " Gelem/s" <<endl;
}

bool CReductionTask::ValidateResults()
{
	bool success = true;

	for (const auto& kernelName : g_kernelNames)
	{
		bool equalData = memcmp(m_resultGPU.data(), m_resultCPU.data(), m_resultGPU.size() * sizeof(DataType)) == 0;
		if (!equalData)
		{
            cout << "Validation of radixsort kernel " << kernelName << " failed.";
			success = false;
		}
	}

	return success;
}

void CReductionTask::RadixSort(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
    cl_int clErr;
    size_t globalWorkSize[1] = { m_N };
    size_t localWorkSize[3];
    memcpy(localWorkSize, LocalWorkSize, 3 * sizeof(size_t));

    clErr = clSetKernelArg(m_kernelMap["RadixSort"], 0, sizeof(cl_mem), (void*)&m_dInputArray);
    clErr = clSetKernelArg(m_kernelMap["RadixSort"], 1, sizeof(cl_mem), (void*)&m_dResultArray);
    clErr |= clEnqueueNDRangeKernel(CommandQueue, m_kernelMap["RadixSort"], 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    clErr |= clFinish(CommandQueue); // kill
    if (clErr != CL_SUCCESS) {
        cerr << __LINE__ << ": Kernel execution failure: " << CLUtil::GetCLErrorString(clErr) << endl;
    }
}

void CReductionTask::RadixSortReadWrite(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	cl_int clErr;
    size_t globalWorkSize[1] = { m_N };
    size_t localWorkSize[3];
    memcpy(localWorkSize, LocalWorkSize, 3 * sizeof(size_t));

    clErr = clSetKernelArg(m_kernelMap["RadixSortReadWrite"], 0, sizeof(cl_mem), (void*)&m_dReadWriteArray);
    clErr |= clEnqueueNDRangeKernel(CommandQueue, m_kernelMap["RadixSortReadWrite"], 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	clErr |= clFinish(CommandQueue); // kill
	if (clErr != CL_SUCCESS) {
		cerr << __LINE__ << ": Kernel execution failure: " << CLUtil::GetCLErrorString(clErr) << endl;
	}
}

void CReductionTask::ExecuteTask(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int alternative)
{
	//write input data to the GPU
	bool blocking = CL_FALSE;
	const size_t offset = 0;
	const size_t dataSize = m_N * sizeof(DataType);
	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dInputArray, blocking, offset, dataSize, m_hInput.data(), 0, NULL, NULL), "Error copying data from host to device!");

    decltype(m_dResultArray) deviceResultArray;

	//run selected task
	switch (alternative) {
		case 0:
			RadixSort(Context, CommandQueue, LocalWorkSize);
            deviceResultArray = m_dResultArray;
			break;
        case 1:
            V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dReadWriteArray, blocking, offset, dataSize, m_hInput.data(), 0, NULL, NULL), "Error copying data from host to device!");
            RadixSortReadWrite(Context, CommandQueue, LocalWorkSize);
            deviceResultArray = m_dReadWriteArray;
            break;
        default:
            V_RETURN_CL(false, "Invalid task selected");
            break;
	}

	//read back the results synchronously.
	std::fill(m_resultGPU.begin(), m_resultGPU.end(), 0);
	blocking = CL_TRUE;
	
    V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, deviceResultArray, blocking, offset, dataSize, m_resultGPU.data(), 0, NULL, NULL), "Error reading data from device!");
}

void CReductionTask::TestPerformance(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int Task)
{
    cout << "Testing performance of task " << g_kernelNames[Task] << endl;

    //write input data to the GPU
    V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dInputArray, CL_FALSE, 0, m_N * sizeof(DataType), m_hInput.data(), 0, NULL, NULL), "Error copying data from host to device!");

    //finish all before we start meassuring the time
    V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");

    CTimer timer;
    timer.Start();

    //run the kernel N times
    unsigned int nIterations = 100;
    for (unsigned int i = 0; i < nIterations; i++) {
        //run selected task
        switch (Task) {
        case 0:
            RadixSort(Context, CommandQueue, LocalWorkSize);
            break;
        case 1:
            V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dReadWriteArray, CL_FALSE, 0, m_N * sizeof(DataType), m_hInput.data(), 0, NULL, NULL), "Error copying data from host to device!");
            RadixSortReadWrite(Context, CommandQueue, LocalWorkSize);
            break;
        }

    }

    //wait until the command queue is empty again
    V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");

    timer.Stop();

    double ms = timer.GetElapsedMilliseconds() / double(nIterations);
    cout << "  average time: " << ms << " ms, throughput: " << 1.0e-6 * (double)m_N / ms << " Gelem/s" << endl;
}

///////////////////////////////////////////////////////////////////////////////
