#include "CRadixSortTask.h"
#include "CRadixSortCPU.h"
#include "RadixSortOptions.h"

#include "../Common/CLUtil.h"
#include "../Common/CTimer.h"
#include "../Common/CLTypeInformation.h"

#include "ComputeDeviceData.h"

#include <algorithm>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ctime>        // std::time_t, struct std::tm, std::localtime
#include <chrono>       // std::chrono::system_clock
#include <functional>
#include <type_traits>
#include <cstring>      // memcmp
#include <cassert>

#include <sys/stat.h>

//#define MORE_PROFILING
template <typename DataType>
CRadixSortTask<DataType>::CRadixSortTask(
    const RadixSortOptions& options,
    std::shared_ptr<Dataset<DataType>> dataset)
	:
    mNumberKeys(static_cast<decltype(mNumberKeys)>(options.num_elements)),
	mNumberKeysRounded(Parameters::_NUM_MAX_INPUT_ELEMS),
	hostData(dataset),
    options(options)
{}

template <typename DataType>
CRadixSortTask<DataType>::~CRadixSortTask()
{
	ReleaseResources();
}

template <typename T>
typename std::enable_if<!std::is_integral<T>::value>::type
appendToOptions(std::string& dst, const std::string& key, const T& obj)
{
    dst += " -D" + key + "=" + "'" + std::string(obj) + "'";
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value>::type
appendToOptions(std::string& dst, const std::string& key, const T& value)
{
    dst += " -D" + key + "=" + "'" + std::to_string(value) + "'";
}


template <typename DataType>
std::string CRadixSortTask<DataType>::buildOptions()
{
    std::string options;
    //options += " -cl-opt-disable";
    options += " -cl-nv-verbose";
    // Compile options string
    {
        ///////////////////////////////////////////////////////
        // these parameters can be changed
        appendToOptions(options, "_ITEMS", Parameters::_NUM_ITEMS_PER_GROUP); // number of items in a group
        appendToOptions(options, "_GROUPS", Parameters::_NUM_GROUPS); // the number of virtual processors is Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS
        appendToOptions(options, "_HISTOSPLIT", Parameters::_NUM_HISTOSPLIT); // number of splits of the histogram
        appendToOptions(options, "_TOTALBITS", Parameters::_TOTALBITS);  // number of bits for the integer in the list (max=32)
        appendToOptions(options, "_BITS", Parameters::_NUM_BITS_PER_RADIX);  // number of bits in the radix
        // max size of the sorted vector
        // it has to be divisible by  Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS
        // (for other sizes, pad the list with big values)
        appendToOptions(options, "_N", Parameters::_NUM_MAX_INPUT_ELEMS);// maximal size of the list
        //#define PERMUT  // store the final permutation
        ////////////////////////////////////////////////////////

        // the following parameters are computed from the previous
        appendToOptions(options, "_RADIX", Parameters::_RADIX);//  radix  = 2^_BITS
        appendToOptions(options, "_PASS", Parameters::_NUM_PASSES); // number of needed passes to sort the list
        appendToOptions(options, "_HISTOSIZE", Parameters::_HISTOSIZE);// size of the histogram
        // maximal value of integers for the sort to be correct
        //appendToOptions(options, "_MAXINT", Parameters::_MAXINT);
    }
    return options;
}

template <typename DataType>
bool CRadixSortTask<DataType>::InitResources(cl_device_id Device, cl_context Context)
{
    // CPU resources

    //for (size_t i = 0; i < Parameters::_NUM_MAX_INPUT_ELEMS; i++) {
    //	//m_hInput[i] = m_N - i;			// Use this for debugging
    //	// Mersienne twister
    //	m_hKeys[i] = dis(generator);
    //	//m_hInput[i] = rand() & 15;
    //}

    //std::copy(m_hInput.begin(),
    //	m_hInput.begin() + 100,
    //	std::ostream_iterator<DataType>(std::cout, "\n"));

    CheckLocalMemory(Device);
    AllocateDeviceMemory(Context);

    //load and compile kernels
    {
        std::string programCode;
        if(!CLUtil::LoadProgramSourceToMemory("RadixSort.cl", programCode)) {
            return false;
        }

        using UnsignedType = typename std::make_unsigned<DataType>::type;

		const auto OFFSET { -std::numeric_limits<DataType>::min() };
        std::stringstream ss;
        ss << "#define DataType " << TypeNameString<DataType>::open_cl_name << std::endl
           << "#define UnsignedDataType " << TypeNameString< UnsignedType >::open_cl_name << std::endl
           << "#define OFFSET " << OFFSET << std::endl
           << programCode << std::endl;

        const auto completeCode = ss.str();
        const auto options = buildOptions();
        deviceData->m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, completeCode, options);
        if (deviceData->m_Program == nullptr) {
            return false;
        }
    }

	cl_int clError;
	// Create each kernel in global kernel list
	hostData.m_hResultGPUMap["RadixSort_01"] = std::vector<DataType>(mNumberKeysRounded);
    for (const auto& kernelName : deviceData->kernelNames) {
		// Input data stays the same for each kernel
        deviceData->m_kernelMap[kernelName] = clCreateKernel(deviceData->m_Program, kernelName.c_str(), &clError);

        const auto errorMsg { std::string("Failed to create kernel: ") + kernelName };
        V_RETURN_FALSE_CL(clError, errorMsg.c_str());
    }

	return true;
}

template <typename DataType>
void CRadixSortTask<DataType>::ReleaseResources()
{
	// free device resources
    // implicitly done by destructor of ComputeDeviceData
}

template <typename DataType>
void CRadixSortTask<DataType>::ComputeGPU(
    cl_context Context,
    cl_command_queue CommandQueue,
    const std::array<size_t,3>& LocalWorkSize)
{
	padGPUData(CommandQueue);
	ExecuteTask(Context, CommandQueue, LocalWorkSize, "RadixSort_01");

	TestPerformance(Context, CommandQueue, LocalWorkSize, 0);
}

template <typename DataType>
void CRadixSortTask<DataType>::ComputeCPU()
{
	std::copy(hostData.m_hKeys.begin(), hostData.m_hKeys.end(), hostData.m_resultSTLCPU.begin());

	CTimer timer;
	timer.Start();
	for (auto j = 0U; j < Parameters::_NUM_PERFORMANCE_ITERATIONS; j++) {
		std::copy(hostData.m_hKeys.begin(), hostData.m_hKeys.begin() + mNumberKeysRounded, hostData.m_resultSTLCPU.begin());

		// Reference sorting (STL quicksort):
		std::sort(hostData.m_resultSTLCPU.begin(), hostData.m_resultSTLCPU.begin() + mNumberKeysRounded);
	}
	timer.Stop();
	cpu_stl_time.avg = timer.GetElapsedMilliseconds() / double(Parameters::_NUM_PERFORMANCE_ITERATIONS);

	hostData.m_resultRadixSortCPU.resize(mNumberKeysRounded);

	timer.Start();
	for (auto j = 0U; j < Parameters::_NUM_PERFORMANCE_ITERATIONS; j++) {
		std::copy(hostData.m_hKeys.begin(), hostData.m_hKeys.begin() + mNumberKeysRounded, hostData.m_resultRadixSortCPU.begin());

		// Reference sorting implementation on CPU (radixsort):
		RadixSortCPU<DataType>::sort(hostData.m_resultRadixSortCPU);
    }
	timer.Stop();

    cpu_radix_time.avg = timer.GetElapsedMilliseconds() / double(Parameters::_NUM_PERFORMANCE_ITERATIONS);
}

template <typename DataType>
bool CRadixSortTask<DataType>::ValidateResults()
{
	bool success = true;

	for (const auto& alternative : deviceData->alternatives)
	{
		const bool validCPURadixSort = memcmp(hostData.m_resultRadixSortCPU.data(), hostData.m_resultSTLCPU.data(), sizeof(DataType) * mNumberKeys) == 0;
		const bool validGPURadixSort = memcmp(hostData.m_hResultGPUMap[alternative].data(), hostData.m_resultSTLCPU.data(), sizeof(DataType) * mNumberKeys) == 0;

		const std::string hasPassedCPU = validCPURadixSort ? "passed" : "FAILED";
		const std::string hasPassedGPU = validGPURadixSort ? "passed" : "FAILED";

		std::cout << "Data set: " << hostData.m_selectedDataset->name() << std::endl;
		std::cout << "Data type: " << TypeNameString<DataType>::stdint_name << std::endl;
		std::cout << "Validation of CPU RadixSort has " + hasPassedCPU << std::endl;
		std::cout << "Validation of GPU RadixSort has " + hasPassedGPU << std::endl;

		success = success && validCPURadixSort && validGPURadixSort;
	}

	return success;
}

template <typename DataType>
void CRadixSortTask<DataType>::Histogram(cl_command_queue CommandQueue, int pass)
{
    const size_t nbitems = Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS;
    const size_t nblocitems = Parameters::_NUM_ITEMS_PER_GROUP;

	assert(mNumberKeysRounded % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);
	assert(mNumberKeysRounded <= Parameters::_NUM_MAX_INPUT_ELEMS);

	const auto histogramKernelHandle = deviceData->m_kernelMap["histogram"];

	// Set kernel arguments
	{
        cl_uint argIdx = 0U;
		V_RETURN_CL(clSetKernelArg(histogramKernelHandle, argIdx++, sizeof(cl_mem), &deviceData->m_dInKeys), "Could not set input elements argument");
		V_RETURN_CL(clSetKernelArg(histogramKernelHandle, argIdx++, sizeof(cl_mem), &deviceData->m_dHistograms), "Could not set input histograms");
		V_RETURN_CL(clSetKernelArg(histogramKernelHandle, argIdx++, sizeof(pass), &pass), "Could not set pass argument");
		V_RETURN_CL(clSetKernelArg(histogramKernelHandle, argIdx++, sizeof(cl_int) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP, NULL), "Could not set local cache");
		V_RETURN_CL(clSetKernelArg(histogramKernelHandle, argIdx++, sizeof(int), &mNumberKeysRounded), "Could not set key count");
	}

    cl_event eve;

    CTimer timer;
    timer.Start();
	// Execute kernel
	V_RETURN_CL(clEnqueueNDRangeKernel(CommandQueue,
		histogramKernelHandle,
        1, NULL,
        &nbitems,
        &nblocitems,
        0, NULL, &eve),
		"Could not execute histogram kernel");

    clFinish(CommandQueue);
    timer.Stop();
    histo_time.update(timer.GetElapsedMilliseconds());

#ifdef MORE_PROFILING
    cl_ulong debut, fin;
    cl_int err;
    err = clGetEventProfilingInfo(eve,
        CL_PROFILING_COMMAND_QUEUED,
        sizeof(cl_ulong),
        (void*)&debut,
        NULL);
    //std::cout << err<<" , "<<CL_PROFILING_INFO_NOT_AVAILABLE<<std::endl;
    assert(err == CL_SUCCESS);

    err = clGetEventProfilingInfo(eve,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        (void*)&fin,
        NULL);
    assert(err == CL_SUCCESS);

    histo_time += (float)(fin - debut) / 1e9f;
#endif
}

template <typename DataType>
void CRadixSortTask<DataType>::ScanHistogram(cl_command_queue CommandQueue, int pass)
{
    static_cast<void>(pass);
    // numbers of processors for the local scan
    // = half the size of the local histograms
    // global work size
	size_t nbitems    = Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP / 2;
    // local work size
	size_t nblocitems = nbitems / Parameters::_NUM_HISTOSPLIT;

	const uint32_t maxmemcache = std::max(Parameters::_NUM_HISTOSPLIT,
		Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS * Parameters::_RADIX / Parameters::_NUM_HISTOSPLIT);

    // scan locally the histogram (the histogram is split into several
    // parts that fit into the local memory)

	const auto scanHistogramKernel  = deviceData->m_kernelMap["scanhistograms"];
    const auto pasteHistogramKernel = deviceData->m_kernelMap["pastehistograms"];
    // Set kernel arguments
    {
        cl_uint argIdx = 0U;
        V_RETURN_CL(clSetKernelArg(scanHistogramKernel, argIdx++, sizeof(cl_mem), &deviceData->m_dHistograms), "Could not set histogram argument");
        V_RETURN_CL(clSetKernelArg(scanHistogramKernel, argIdx++, sizeof(uint32_t) * maxmemcache, NULL), "Could not set histogram cache size"); // mem cache
        V_RETURN_CL(clSetKernelArg(scanHistogramKernel, argIdx++, sizeof(cl_mem), &deviceData->m_dGlobsum), "Could not set global histogram argument");
    }
    cl_event eve;

	// Execute kernel for first scan (local)
    const cl_uint workDimension = 1;
    size_t* globalWorkOffset = nullptr;

    CTimer timer;
    timer.Start();
    V_RETURN_CL(clEnqueueNDRangeKernel(CommandQueue,
		scanHistogramKernel,
        workDimension,
        globalWorkOffset,
        &nbitems,
        &nblocitems,
        0, NULL, &eve), "Could not execute 1st instance of scanHistogram kernel.");

    clFinish(CommandQueue);
    timer.Stop();
    scan_time.update(timer.GetElapsedMilliseconds());

#ifdef MORE_PROFILING
    cl_int err = CL_SUCCESS;
    cl_ulong debut, fin;

    err = clGetEventProfilingInfo(eve,
        CL_PROFILING_COMMAND_QUEUED,
        sizeof(cl_ulong),
        (void*)&debut,
        NULL);
    assert(err == CL_SUCCESS);

    err = clGetEventProfilingInfo(eve,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        (void*)&fin,
        NULL);
    assert(err == CL_SUCCESS);

    scan_time += (float)(fin - debut) / 1e9f;
#endif

    // second scan for the globsum
    // Set kernel arguments
    {
        V_RETURN_CL(
            clSetKernelArg(scanHistogramKernel,
            0,
            sizeof(cl_mem),
            &deviceData->m_dGlobsum),
            "Could not set global sum parameter");
        V_RETURN_CL(clSetKernelArg(scanHistogramKernel,
                    2,
                    sizeof(cl_mem),
                    &deviceData->m_dTemp),
                "Could not set temporary parameter");
    }

    // global work size
	nbitems    = Parameters::_NUM_HISTOSPLIT / 2;
    // local work size
    nblocitems = nbitems;

    timer.Start();
	// Execute kernel for second scan (global)
    V_RETURN_CL(clEnqueueNDRangeKernel(CommandQueue,
		scanHistogramKernel,
        workDimension,
        globalWorkOffset,
        &nbitems,
        &nblocitems,
        0, NULL, &eve),
    "Could not execute 2nd instance of scanHistogram kernel.");

    clFinish(CommandQueue);
    timer.Stop();
    scan_time.update(timer.GetElapsedMilliseconds());


#ifdef MORE_PROFILING
    err = clGetEventProfilingInfo(eve,
        CL_PROFILING_COMMAND_QUEUED,
        sizeof(cl_ulong),
        (void*)&debut,
        NULL);
    assert(err == CL_SUCCESS);

    err = clGetEventProfilingInfo(eve,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        (void*)&fin,
        NULL);
    assert(err == CL_SUCCESS);

    scan_time += static_cast<float>(fin - debut) / 1e9f;
#endif

    // loops again in order to paste together the local histograms
    // global
	nbitems    = Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP / 2;
    // local work size
	nblocitems = nbitems / Parameters::_NUM_HISTOSPLIT;

	V_RETURN_CL(clSetKernelArg(pasteHistogramKernel, 0, sizeof(cl_mem), &deviceData->m_dHistograms), "Could not set histograms argument");
	V_RETURN_CL(clSetKernelArg(pasteHistogramKernel, 1, sizeof(cl_mem), &deviceData->m_dGlobsum), "Could not set globsum argument");

	// Execute paste histogram kernel
    timer.Start();
    V_RETURN_CL(clEnqueueNDRangeKernel(CommandQueue,
		pasteHistogramKernel,
        workDimension,
        globalWorkOffset,
        &nbitems,
        &nblocitems,
        0, NULL, &eve), "Could not execute paste histograms kernel");

    clFinish(CommandQueue);
    timer.Stop();
    paste_time.update(timer.GetElapsedMilliseconds());

#ifdef MORE_PROFILING
    err = clGetEventProfilingInfo(eve,
        CL_PROFILING_COMMAND_QUEUED,
        sizeof(cl_ulong),
        (void*)&debut,
        NULL);
    assert(err == CL_SUCCESS);

    err = clGetEventProfilingInfo(eve,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        (void*)&fin,
        NULL);
    assert(err == CL_SUCCESS);

    scan_time += (float)(fin - debut) / 1e9f;
#endif
}

template <typename DataType>
void CRadixSortTask<DataType>::Reorder(cl_command_queue CommandQueue, int pass)
{
	const size_t nblocitems = Parameters::_NUM_ITEMS_PER_GROUP;
    const size_t nbitems    = Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS;

	assert(mNumberKeysRounded % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);

    clFinish(CommandQueue);
    auto reorderKernel = deviceData->m_kernelMap["reorder"];

	//  const __global int* d_inKeys,
	//  __global int* d_outKeys,
	//	__global int* d_Histograms,
	//	const int pass,
	//	__global int* d_inPermut,
	//	__global int* d_outPermut,
	//	__local int* loc_histo,
	//	const int n

	struct ReorderKernelParams {
		cl_mem inKeys;
		cl_mem outKeys;
		cl_mem histograms;
		int pass;
		cl_mem inPermutation;
		cl_mem outPermutation;
		size_t localHistogramSize;
		int numElems;
	};

	// set kernel arguments
	{
        cl_uint argIdx = 0U;
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(cl_mem), &deviceData->m_dInKeys), "Could not set input keys for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(cl_mem), &deviceData->m_dOutKeys), "Could not set output keys for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(cl_mem), &deviceData->m_dHistograms), "Could not set histograms for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(pass),   &pass), "Could not set pass for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(cl_mem), &deviceData->m_dInPermut), "Could not set input permutation for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(cl_mem), &deviceData->m_dOutPermut), "Could not set output permutation for reorder kernel.");
		V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++,
			sizeof(cl_int) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP,
            NULL), "Could not set local memory for reorder kernel."); // mem cache

        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(mNumberKeysRounded), &mNumberKeysRounded), "Could not set number of input keys for reorder kernel.");
	}

	assert(Parameters::_RADIX == pow(2, Parameters::_NUM_BITS_PER_RADIX));

    cl_event eve;

    const cl_uint workDimension = 1;
    const size_t* globalWorkOffset = nullptr;
	// Execute kernel
    CTimer timer;
    timer.Start();
	V_RETURN_CL(clEnqueueNDRangeKernel(CommandQueue,
		reorderKernel,
        workDimension,
        globalWorkOffset,
		&nbitems,
		&nblocitems,
		0, NULL, &eve), "Could not execute reorder kernel");
    clFinish(CommandQueue);
    timer.Stop();
    reorder_time.update(timer.GetElapsedMilliseconds());

#ifdef MORE_PROFILING
    cl_int err = CL_SUCCESS;
    cl_ulong debut, fin;

    err = clGetEventProfilingInfo(eve,
        CL_PROFILING_COMMAND_QUEUED,
        sizeof(cl_ulong),
        (void*)&debut,
        NULL);
    assert(err == CL_SUCCESS);

    err = clGetEventProfilingInfo(eve,
        CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),
        (void*)&fin,
        NULL);
    assert(err == CL_SUCCESS);

    reorder_time += (float)(fin - debut) / 1e9f;
#endif

    // swap the old and new vectors of keys
    std::swap(deviceData->m_dInKeys, deviceData->m_dOutKeys);

    // swap the old and new permutations
    std::swap(deviceData->m_dInPermut, deviceData->m_dOutPermut);
}

/// Check divisibility of works to assign correct amounts of work to groups/work-items.
template <typename DataType>
void CRadixSortTask<DataType>::CheckDivisibility()
{
    static_assert(Parameters::_RADIX == 1 << Parameters::_NUM_BITS_PER_RADIX);
    static_assert(Parameters::_TOTALBITS % Parameters::_NUM_BITS_PER_RADIX == 0);
    static_assert(Parameters::_NUM_MAX_INPUT_ELEMS % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);
    static_assert((Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_RADIX) % Parameters::_NUM_HISTOSPLIT == 0);
}

template <typename DataType>
void CRadixSortTask<DataType>::CheckLocalMemory(cl_device_id Device)
{
    // check that the local mem is sufficient (suggestion of Jose Luis Cerc\F3s Pita)
    cl_ulong localMem;
	clGetDeviceInfo(Device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, NULL);
    if (options.verbose) {
        std::cout << "Cache size   = " << localMem << " Bytes." << std::endl;
		std::cout << "Needed cache = " << sizeof(cl_uint) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP << " Bytes." << std::endl;
    }
	assert(localMem > sizeof(DataType) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP);

	unsigned int maxmemcache = std::max(Parameters::_NUM_HISTOSPLIT,
		Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS * Parameters::_RADIX / Parameters::_NUM_HISTOSPLIT);
	assert(localMem > sizeof(DataType)*maxmemcache);
}

/// resize the sorted vector
template <typename DataType>
void CRadixSortTask<DataType>::Resize(uint32_t nn)
{
	assert(nn <= Parameters::_NUM_MAX_INPUT_ELEMS);

    if (options.verbose){
        std::cout << "Resize to  " << nn << std::endl;
    }
    mNumberKeys = nn;

    // length of the vector has to be divisible by (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP)
    const int32_t rest = mNumberKeys % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP);
    mNumberKeysRounded = mNumberKeys;

    if (rest != 0) {
		mNumberKeysRounded = mNumberKeys - rest + (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP);
    }
	mNumberKeysRest = rest;
}

template <typename DataType>
void CRadixSortTask<DataType>::padGPUData(cl_command_queue CommandQueue)
{
	if (mNumberKeysRest != 0) {
		const auto MAX_INT = std::numeric_limits<DataType>::max();
		// pad the vector with big values
		const std::vector<DataType> pad(
            Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP,
            MAX_INT - 1);

		assert(mNumberKeysRounded <= Parameters::_NUM_MAX_INPUT_ELEMS);

		const auto blocking = CL_TRUE;
		const auto offset = sizeof(DataType) * mNumberKeys;
		const auto size = sizeof(DataType) * (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP - mNumberKeysRest);
		V_RETURN_CL(clEnqueueWriteBuffer(
            CommandQueue,
			deviceData->m_dInKeys,
			blocking,
			offset,
			size,
			pad.data(),
			0, NULL, NULL),
        "Could not write input data");
	}
}

template <typename DataType>
void CRadixSortTask<DataType>::RadixSort(cl_context Context, cl_command_queue CommandQueue, const std::array<size_t,3>& LocalWorkSize)
{
    CheckDivisibility();

    static_cast<void>(Context);
    static_cast<void>(LocalWorkSize);

	assert(mNumberKeysRounded <= Parameters::_NUM_MAX_INPUT_ELEMS);
    assert(mNumberKeys <= mNumberKeysRounded);

    if (options.verbose) {
        std::cout << "Start sorting " << mNumberKeys << " keys." << std::endl;
    }

    for (uint32_t pass = 0; pass < Parameters::_NUM_PASSES; pass++){
        if (options.verbose) {
            std::cout << "Pass " << pass << ":" << std::endl;
        }

        if (options.verbose) {
            std::cout << "Building histograms" << std::endl;
        }
        Histogram(CommandQueue, pass);

        if (options.verbose) {
            std::cout << "Scanning histograms" << std::endl;
        }
        ScanHistogram(CommandQueue, pass);

        if (options.verbose) {
            std::cout << "Reordering " << std::endl;
        }
        Reorder(CommandQueue, pass);

        if (options.verbose) {
            std::cout << "-------------------" << std::endl;
        }
    }

    sort_time.avg = histo_time.avg + scan_time.avg + reorder_time.avg + paste_time.avg;
    sort_time.n = histo_time.n;
    if (options.verbose){
        std::cout << "End sorting" << std::endl;
    }
}

template <typename DataType>
void CRadixSortTask<DataType>::AllocateDeviceMemory(cl_context Context)
{
	// Done in constructor of ComputeDeviceData :)
	Resize(mNumberKeys);
	deviceData = std::make_shared<ComputeDeviceData<DataType>>(Context, mNumberKeysRounded);
}

template <typename DataType>
void CRadixSortTask<DataType>::CopyDataToDevice(cl_command_queue CommandQueue)
{
	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue,
        deviceData->m_dInKeys,
        CL_TRUE, 0,
        sizeof(DataType) * mNumberKeysRounded,
        hostData.m_hKeys.data(),
        0, NULL, NULL),
		"Could not initialize input keys device buffer");

    clFinish(CommandQueue);  // wait end of read

	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue,
        deviceData->m_dInPermut,
        CL_TRUE, 0,
        sizeof(uint32_t) * mNumberKeysRounded,
        hostData.h_Permut.data(),
        0, NULL, NULL),
		"Could not initialize input permutation device buffer");

    clFinish(CommandQueue);  // wait end of read
}

template <typename DataType>
void CRadixSortTask<DataType>::CopyDataFromDevice(cl_command_queue CommandQueue)
{
	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue,
        deviceData->m_dInKeys,
		CL_TRUE, 0,
		sizeof(DataType) * mNumberKeysRounded,
        hostData.m_hResultGPUMap["RadixSort_01"].data(),
		0, NULL, NULL),
		"Could not read result data");

	clFinish(CommandQueue);  // wait end of read

	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue,
        deviceData->m_dInPermut,
		CL_TRUE, 0,
		sizeof(uint32_t)  * mNumberKeysRounded,
        hostData.h_Permut.data(),
		0, NULL, NULL),
		"Could not read result permutation");

	clFinish(CommandQueue);  // wait end of read

	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue,
        deviceData->m_dHistograms,
		CL_TRUE, 0,
		sizeof(uint32_t)  * Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP,
        hostData.m_hHistograms.data(),
		0, NULL, NULL),
		"Could not read result histograms");

	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue,
        deviceData->m_dGlobsum,
		CL_TRUE, 0,
		sizeof(uint32_t)  * Parameters::_NUM_HISTOSPLIT,
		hostData.m_hGlobsum.data(),
		0, NULL, NULL),
		"Could not read result global sum");

	clFinish(CommandQueue);  // wait end of read
}

template <typename DataType>
void CRadixSortTask<DataType>::ExecuteTask(cl_context Context, cl_command_queue CommandQueue, const std::array<size_t,3>& LocalWorkSize, const std::string& alternative)
{
	//run selected task
	if (alternative == "RadixSort_01") {
        CopyDataToDevice(CommandQueue);
		RadixSort(Context, CommandQueue, LocalWorkSize);
		CopyDataFromDevice(CommandQueue);
	} else {
		V_RETURN_CL(false, "Invalid task selected");
	}
}

template <typename DataType>
template <typename Stream>
void CRadixSortTask<DataType>::writePerformance(Stream&& stream)
{
    const std::vector<std::string> columns {
        "NumElements", "Datatype", "Dataset", "avgHistogram", "avgScan", "avgPaste", "avgReorder", "avgTotalGPU", "avgTotalSTLCPU", "avgTotalRDXCPU"
    };

    stream << columns[0];
    for (auto i = 1U; i < columns.size(); i++) {
        stream << "," << columns[i];
    }

    stream << std::endl;
    stream << mNumberKeys << ",";
    stream << TypeNameString<DataType>::stdint_name << ",";
    stream << hostData.m_selectedDataset->name() << ",";
    stream << histo_time.avg << ",";
    stream << scan_time.avg << ",";
    stream << paste_time.avg << ",";
    stream << reorder_time.avg << ",";
    stream << sort_time.avg << ",";
    stream << cpu_stl_time.avg << ",";
    stream << cpu_radix_time.avg;
    stream << std::endl;
}

template <typename DataType>
void CRadixSortTask<DataType>::TestPerformance(cl_context Context, cl_command_queue CommandQueue, const std::array<size_t,3>& LocalWorkSize, unsigned int Task)
{
    if (options.perf_to_stdout) {
        std::cout << " radixsort cpu avg time: " << cpu_radix_time.avg << " ms, throughput: " << 1.0e-6 * (double)mNumberKeysRounded / cpu_radix_time.avg << " Gelem/s" << std::endl;
        std::cout << " stl cpu avg time: " << cpu_stl_time.avg << " ms, throughput: " << 1.0e-6 * (double)mNumberKeysRounded / cpu_stl_time.avg << " Gelem/s" << std::endl;
    }

    if (options.perf_to_stdout) {
        std::cout << "Testing performance of GPU task " << deviceData->kernelNames[Task] << std::endl;
    }

    //finish all before we start measuring the time
    V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");

    CTimer timer;
    timer.Start();

    //run the kernel N times
	for (auto i = 0U; i < Parameters::_NUM_PERFORMANCE_ITERATIONS; i++) {
        //run selected task
        switch (Task) {
        case 0:
            CopyDataToDevice(CommandQueue);
            RadixSort(Context, CommandQueue, LocalWorkSize);
            CopyDataFromDevice(CommandQueue);
            break;
        }
    }

    //wait until the command queue is empty again
    V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");

    timer.Stop();
	double ms = timer.GetElapsedMilliseconds() / double(Parameters::_NUM_PERFORMANCE_ITERATIONS);
    sort_time.avg = ms;
    if (options.perf_to_stdout) {

        std::cout << " kernel |    avg      |     min     |    max " << std::endl;
        std::cout << " -----------------------------------------------" << std::endl;
        std::cout << "  histogram: " << std::setw(8) << histo_time.avg << " | " << histo_time.min << " | " << histo_time.max << std::endl;
        std::cout << "  scan:      " << std::setw(8) << scan_time.avg << " | " << scan_time.min << " | " << scan_time.max << std::endl;
        std::cout << "  paste:     " << std::setw(8) << paste_time.avg << " | " << paste_time.min << " | " << paste_time.max << std::endl;
        std::cout << "  reorder:   " << std::setw(8) << reorder_time.avg << " | " << reorder_time.min << " | " << reorder_time.max << std::endl;
        std::cout << " -----------------------------------------------" << std::endl;
        std::cout << "  total:     " << ms << " ms, throughput: " << 1.0e-6 * (double)mNumberKeysRounded / ms << " Gelem/s" << std::endl;
    }

    using std::chrono::system_clock;
    std::time_t tt = system_clock::to_time_t(system_clock::now());
    struct std::tm * ptm = std::localtime(&tt);
    const std::string dateFormat = "%H-%M-%S";
    std::stringstream fileNameBuilder;

    fileNameBuilder << "radix_" << std::put_time(ptm, dateFormat.c_str()) << ".csv";

    if (options.perf_to_csv) {
        const auto filename = fileNameBuilder.str();
        bool file_exists = false;

        {
            struct stat buffer;
            file_exists = (stat(filename.c_str(), &buffer) == 0);
        }

        // Print columns
        if (file_exists)
        {
            std::cout << "File " << filename << " already exists, not overwriting!" << std::endl;
        } else {
            std::ofstream outstream(filename, std::ofstream::out | std::ofstream::app);
            writePerformance(outstream);
        }
    }
    if (options.perf_csv_to_stdout) {
        writePerformance(std::cout);
    }
}

// Specialize CRadixSortTask for exactly these four types.
template class CRadixSortTask < int32_t >;
template class CRadixSortTask < int64_t >;
template class CRadixSortTask < uint32_t >;
template class CRadixSortTask < uint64_t >;
