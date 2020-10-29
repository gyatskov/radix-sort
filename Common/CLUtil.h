#ifndef CL_UTIL_H
#define CL_UTIL_H

#include "CommonDefs.h"

// All OpenCL headers
#if defined(WIN32)
    #include <CL/opencl.h>
#elif defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

#include <string>
#include <iostream>
#include <algorithm>

//! Utility class for frequently-needed OpenCL tasks
class CLUtil
{
public:
	//! Determines the OpenCL global work size given the number of data elements and threads per workgroup
	static size_t GetGlobalWorkSize(size_t DataElemCount, size_t LocalWorkSize);

	//! Loads a program source to memory as a string
	static bool LoadProgramSourceToMemory(const std::string& Path, std::string& SourceCode);

	//! Builds a CL program
	static cl_program BuildCLProgramFromMemory(
        cl_device_id Device,
        cl_context Context,
        const std::string& SourceCode,
        const std::string& options = ""
    );

	static void PrintBuildLog(cl_program Program, cl_device_id Device);

	//! Measures the execution time of a kernel by executing it N times and returning the average time in milliseconds.
	/*!
		The scheduling cost of the kernel can be amortized if we enqueue
		the kernel multiple times. If your kernel is simple and fast, use a high number of iterations!
	*/
	static double ProfileKernel(cl_command_queue CommandQueue, cl_kernel Kernel, cl_uint Dimensions,
		const size_t* pGlobalWorkSize, const size_t* pLocalWorkSize, int NIterations);

    #define CL_ERROR(x) case (x): return #x;
	static constexpr const char* GetCLErrorString(cl_int CLErrorCode)
    {
        switch(CLErrorCode)
        {
            CL_ERROR(CL_SUCCESS);
            CL_ERROR(CL_DEVICE_NOT_FOUND);
            CL_ERROR(CL_DEVICE_NOT_AVAILABLE);
            CL_ERROR(CL_COMPILER_NOT_AVAILABLE);
            CL_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
            CL_ERROR(CL_OUT_OF_RESOURCES);
            CL_ERROR(CL_OUT_OF_HOST_MEMORY);
            CL_ERROR(CL_PROFILING_INFO_NOT_AVAILABLE);
            CL_ERROR(CL_MEM_COPY_OVERLAP);
            CL_ERROR(CL_IMAGE_FORMAT_MISMATCH);
            CL_ERROR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
            CL_ERROR(CL_BUILD_PROGRAM_FAILURE);
            CL_ERROR(CL_MAP_FAILURE);
            CL_ERROR(CL_INVALID_VALUE);
            CL_ERROR(CL_INVALID_DEVICE_TYPE);
            CL_ERROR(CL_INVALID_PLATFORM);
            CL_ERROR(CL_INVALID_DEVICE);
            CL_ERROR(CL_INVALID_CONTEXT);
            CL_ERROR(CL_INVALID_QUEUE_PROPERTIES);
            CL_ERROR(CL_INVALID_COMMAND_QUEUE);
            CL_ERROR(CL_INVALID_HOST_PTR);
            CL_ERROR(CL_INVALID_MEM_OBJECT);
            CL_ERROR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
            CL_ERROR(CL_INVALID_IMAGE_SIZE);
            CL_ERROR(CL_INVALID_SAMPLER);
            CL_ERROR(CL_INVALID_BINARY);
            CL_ERROR(CL_INVALID_BUILD_OPTIONS);
            CL_ERROR(CL_INVALID_PROGRAM);
            CL_ERROR(CL_INVALID_PROGRAM_EXECUTABLE);
            CL_ERROR(CL_INVALID_KERNEL_NAME);
            CL_ERROR(CL_INVALID_KERNEL_DEFINITION);
            CL_ERROR(CL_INVALID_KERNEL);
            CL_ERROR(CL_INVALID_ARG_INDEX);
            CL_ERROR(CL_INVALID_ARG_VALUE);
            CL_ERROR(CL_INVALID_ARG_SIZE);
            CL_ERROR(CL_INVALID_KERNEL_ARGS);
            CL_ERROR(CL_INVALID_WORK_DIMENSION);
            CL_ERROR(CL_INVALID_WORK_GROUP_SIZE);
            CL_ERROR(CL_INVALID_WORK_ITEM_SIZE);
            CL_ERROR(CL_INVALID_GLOBAL_OFFSET);
            CL_ERROR(CL_INVALID_EVENT_WAIT_LIST);
            CL_ERROR(CL_INVALID_EVENT);
            CL_ERROR(CL_INVALID_OPERATION);
            CL_ERROR(CL_INVALID_GL_OBJECT);
            CL_ERROR(CL_INVALID_BUFFER_SIZE);
            CL_ERROR(CL_INVALID_MIP_LEVEL);
            default:
                return "Unknown error code";
        }
    }
    #undef CL_ERROR
};

// Some useful shortcuts for handling pointers and validating function calls
#define V_RETURN_FALSE_CL(expr, errmsg) {cl_int e=(expr);if(CL_SUCCESS!=e){std::cerr<<"Error: "<<errmsg<<" ["<<CLUtil::GetCLErrorString(e)<<"]"<<std::endl; return false; }}
#define V_RETURN_0_CL(expr, errmsg) {cl_int e=(expr);if(CL_SUCCESS!=e){std::cerr<<"Error: "<<errmsg<<" ["<<CLUtil::GetCLErrorString(e)<<"]"<<std::endl; return 0; }}
#define V_RETURN_CL(expr, errmsg) {cl_int e=(expr);if(CL_SUCCESS!=e){std::cerr<<"Error: "<<errmsg<<" ["<<CLUtil::GetCLErrorString(e)<<"]"<<std::endl; return; }}

#define SAFE_DELETE(ptr) {if(ptr){ delete ptr; ptr = NULL; }}
#define SAFE_DELETE_ARRAY(x) {if(x){delete [] x; x = NULL;}}

#define SAFE_RELEASE_KERNEL(ptr) {if(ptr){ clReleaseKernel(ptr); ptr = NULL; }}
#define SAFE_RELEASE_PROGRAM(ptr) {if(ptr){ clReleaseProgram(ptr); ptr = NULL; }}
#define SAFE_RELEASE_MEMOBJECT(ptr) {if(ptr){ clReleaseMemObject(ptr); ptr = NULL; }}

#endif // CL_UTIL_H
