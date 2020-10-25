#include "CLUtil.h"
#include "CTimer.h"

#include <iostream>
#include <fstream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CLUtil

size_t CLUtil::GetGlobalWorkSize(size_t DataElemCount, size_t LocalWorkSize)
{
	size_t r = DataElemCount % LocalWorkSize;
	if(r == 0)
		return DataElemCount;
	else
		return DataElemCount + LocalWorkSize - r;
}

bool CLUtil::LoadProgramSourceToMemory(const std::string& Path, std::string& SourceCode)
{
	ifstream sourceFile;

	sourceFile.open(Path.c_str());
	if (!sourceFile.is_open())
	{
		cerr << "Failed to open file '" << Path << "'." << endl;
		return false;
	}

	// read the entire file into a string
	sourceFile.seekg(0, ios::end);
	ifstream::pos_type fileSize = sourceFile.tellg();
	sourceFile.seekg(0, ios::beg);

	SourceCode.resize((size_t)fileSize);
	sourceFile.read(&SourceCode[0], fileSize);

	return true;
}

cl_program CLUtil::BuildCLProgramFromMemory(cl_device_id Device, cl_context Context, const std::string& SourceCode, const std::string& options)
{
	cl_program prog;

	// if this macro is defined, we also insert it to all OpenCL kernels
	const string srcSolution = string("#define GPUC_SOLUTION\n\n") + SourceCode;
	const char* src = srcSolution.c_str();
	size_t length = srcSolution.size();

	cl_int clError;
	prog = clCreateProgramWithSource(Context, 1, &src, &length, &clError);
	if(CL_SUCCESS != clError)
	{
		cerr<<"Failed to create CL program from source.";
		return nullptr;
	}

	// program created, now build it:
	clError = clBuildProgram(prog, 1, &Device, options.c_str(), NULL, NULL);
	PrintBuildLog(prog, Device);
	if(CL_SUCCESS != clError)
	{
		cerr<<"Failed to build CL program.";
		SAFE_RELEASE_PROGRAM(prog);
		return nullptr;
	}

	return prog;
}

void CLUtil::PrintBuildLog(cl_program Program, cl_device_id Device)
{
    {
        cl_build_status buildStatus;
        clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL);
        if(buildStatus != CL_SUCCESS) {
            cout<<"OpenCL kernel build failed!"<<endl;
        }
    }

    // first, query size
	size_t logSize;
	clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

    // then, allocate actual memory
	string buildLog(logSize, ' ');

	clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);
	buildLog[logSize] = '\0';

	cout << "Build log:" <<endl;
	cout << '\'' << buildLog << '\'' << endl;
}

double CLUtil::ProfileKernel(
        cl_command_queue CommandQueue,
        cl_kernel Kernel,
        cl_uint Dimensions,
		const size_t* pGlobalWorkSize,
        const size_t* pLocalWorkSize,
        int NIterations)
{
	CTimer timer;
	cl_int clErr;

	// wait until the command queue is empty...
	// Should not be used in production code, but this synchronizes HOST and DEVICE
	clErr = clFinish(CommandQueue);

	timer.Start();

	// run the kernel N times for better average accuracy
	for(int i = 0; i < NIterations; i++)
	{
        constexpr auto pGlobalWorkOffset = nullptr;
        constexpr auto pEventsInWaitList = nullptr;
        constexpr auto pEvent = nullptr;
        constexpr auto numEventsInWaitList = 0U;
		clErr |= clEnqueueNDRangeKernel(
            CommandQueue,
            Kernel,
            Dimensions,
            pGlobalWorkOffset,
            pGlobalWorkSize,
            pLocalWorkSize,
            numEventsInWaitList,
            pEventsInWaitList,
            pEvent
        );
	}
	// wait again to sync
	clErr |= clFinish(CommandQueue);

	timer.Stop();

	if(clErr != CL_SUCCESS)
	{
		const string errorString {GetCLErrorString(clErr)};
		cerr<<"Kernel execution failure: "<<errorString<<endl;
	}

	return timer.GetElapsedMilliseconds() / double(NIterations);
}


///////////////////////////////////////////////////////////////////////////////
