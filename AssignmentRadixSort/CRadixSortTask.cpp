#include "CRadixSortTask.h"
#include "CRadixSortCPU.h"

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
#include <cassert>
#include <sys/stat.h>

using namespace std;

//#define MORE_PROFILING
template <typename DataType>
CRadixSortTask<DataType>::CRadixSortTask(const RadixSortOptions& options, std::shared_ptr<Dataset<DataType>> dataset)
	:
    nkeys(static_cast<decltype(nkeys)>(options.num_elements)),
	nkeys_rounded(Parameters::_NUM_MAX_INPUT_ELEMS),
	hostData(dataset),
    options(options)
{}

template <typename DataType>
CRadixSortTask<DataType>::~CRadixSortTask() 
{
	ReleaseResources();
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value>::type
    appendToOptions(std::string& dst, const std::string& key, const T value) 
{
    dst += " -D" + key + "=" + to_string(value);
}

template <typename T>
typename std::enable_if<!std::is_integral<T>::value>::type
    appendToOptions(std::string& dst, const std::string& key, const T str) 
{
    dst += " -D" + key + "=" + string(str);
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
        
        //appendToOptions(options, "DataType", TypeNameString<DataType>::open_cl_name);
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
        string programCode;
        string dataTypeDefine = "#define DataType " + std::string(TypeNameString<DataType>::open_cl_name) + std::string("\n");
		string unsignedDataTypeDefine = "#define UnsignedDataType " + std::string(TypeNameString< std::make_unsigned<DataType>::type >::open_cl_name) + std::string("\n");
		const auto OFFSET = -std::numeric_limits<DataType>::min();
		string summandDefine = "#define OFFSET " + std::to_string(OFFSET) + std::string("\n");
        size_t programSize = 0;
        CLUtil::LoadProgramSourceToMemory("RadixSort.cl", programCode);
		programCode = dataTypeDefine + unsignedDataTypeDefine + summandDefine + programCode;
        const auto options = buildOptions();
        deviceData->m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode, options);
        if (deviceData->m_Program == nullptr) {
            return false;
        }
    }

	cl_int clError;
	// Create each kernel in global kernel list
	hostData.m_hResultGPUMap["RadixSort_01"] = std::vector<DataType>(nkeys_rounded);
    for (const auto& kernelName : deviceData->kernelNames) {
		// Input data stays the same for each kernel
        deviceData->m_kernelMap[kernelName] = clCreateKernel(deviceData->m_Program, kernelName.c_str(), &clError);
		
        std::string errorMsg("Failed to create kernel: ");
        errorMsg += kernelName;
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
void CRadixSortTask<DataType>::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]) 
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
	for (auto j = 0; j < Parameters::_NUM_PERFORMANCE_ITERATIONS; j++) {
		std::copy(hostData.m_hKeys.begin(), hostData.m_hKeys.begin() + nkeys_rounded, hostData.m_resultSTLCPU.begin());

		// Reference sorting (STL quicksort):
		std::sort(hostData.m_resultSTLCPU.begin(), hostData.m_resultSTLCPU.begin() + nkeys_rounded);
	}
	timer.Stop();
	cpu_stl_time.avg = timer.GetElapsedMilliseconds() / double(Parameters::_NUM_PERFORMANCE_ITERATIONS);

	hostData.m_resultRadixSortCPU.resize(nkeys_rounded);
	
	timer.Start();
	for (auto j = 0; j < Parameters::_NUM_PERFORMANCE_ITERATIONS; j++) {
		std::copy(hostData.m_hKeys.begin(), hostData.m_hKeys.begin() + nkeys_rounded, hostData.m_resultRadixSortCPU.begin());

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
//#define RADIXSORT_CL_NOT_YET_IMPLEMENTED
#ifdef RADIXSORT_CL_NOT_YET_IMPLEMENTED
		std::sort(m_hResultGPUMap[kernelName].begin(), m_hResultGPUMap[kernelName].end());
#endif
		const bool validCPURadixSort = memcmp(hostData.m_resultRadixSortCPU.data(), hostData.m_resultSTLCPU.data(), sizeof(DataType) * nkeys) == 0;
		const bool validGPURadixSort = memcmp(hostData.m_hResultGPUMap[alternative].data(), hostData.m_resultSTLCPU.data(), sizeof(DataType) * nkeys) == 0;

		const std::string hasPassedCPU = validCPURadixSort ? "passed :)" : "FAILED >:O";
		const std::string hasPassedGPU = validGPURadixSort ? "passed :)" : "FAILED >:O";
		cout << "Data set: " << hostData.m_selectedDataset->getName() << std::endl;
		cout << "Data type: " << TypeNameString<DataType>::stdint_name << std::endl;
		cout << "Validation of CPU RadixSort has " + hasPassedCPU << std::endl;
		cout << "Validation of GPU RadixSort has " + hasPassedGPU << std::endl;

		success = success && validCPURadixSort && validGPURadixSort;
	}

	return success;
}

template <typename DataType>
void CRadixSortTask<DataType>::Histogram(cl_command_queue CommandQueue, int pass) 
{
    size_t nbitems = Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS;
    size_t nblocitems = Parameters::_NUM_ITEMS_PER_GROUP;

	assert(nkeys_rounded % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);
	assert(nkeys_rounded <= Parameters::_NUM_MAX_INPUT_ELEMS);

	auto histogramKernel = deviceData->m_kernelMap["histogram"];

	// Set kernel arguments
	{
		V_RETURN_CL(clSetKernelArg(histogramKernel, 0, sizeof(cl_mem), &deviceData->m_dInKeys), "Could not set input elements argument");
		V_RETURN_CL(clSetKernelArg(histogramKernel, 1, sizeof(cl_mem), &deviceData->m_dHistograms), "Could not set input histograms");
		V_RETURN_CL(clSetKernelArg(histogramKernel, 2, sizeof(pass), &pass), "Could not set pass argument");
		V_RETURN_CL(clSetKernelArg(histogramKernel, 3, sizeof(cl_int) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP, NULL), "Could not set local cache");
		V_RETURN_CL(clSetKernelArg(histogramKernel, 4, sizeof(int), &nkeys_rounded), "Could not set key count");
	}

    cl_event eve;

    CTimer timer;
    timer.Start();
	// Execute kernel
	V_RETURN_CL(clEnqueueNDRangeKernel(CommandQueue,
		histogramKernel,
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
    //cout << err<<" , "<<CL_PROFILING_INFO_NOT_AVAILABLE<<endl;
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
    // numbers of processors for the local scan
    // = half the size of the local histograms
    // global work size
	size_t nbitems    = Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP / 2;
    // local work size
	size_t nblocitems = nbitems / Parameters::_NUM_HISTOSPLIT;

	const uint32_t maxmemcache = max(Parameters::_NUM_HISTOSPLIT, 
		Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS * Parameters::_RADIX / Parameters::_NUM_HISTOSPLIT);

    // scan locally the histogram (the histogram is split into several
    // parts that fit into the local memory)

	auto scanHistogramKernel  = deviceData->m_kernelMap["scanhistograms"];
    auto pasteHistogramKernel = deviceData->m_kernelMap["pastehistograms"];
    // Set kernel arguments
    {
        V_RETURN_CL(clSetKernelArg(scanHistogramKernel, 0, sizeof(cl_mem), &deviceData->m_dHistograms), "Could not set histogram argument");
        V_RETURN_CL(clSetKernelArg(scanHistogramKernel, 1, sizeof(uint32_t) * maxmemcache, NULL), "Could not set histogram cache size"); // mem cache
        V_RETURN_CL(clSetKernelArg(scanHistogramKernel, 2, sizeof(cl_mem), &deviceData->m_dGlobsum), "Could not set global histogram argument");
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
        V_RETURN_CL(clSetKernelArg(scanHistogramKernel, 0, sizeof(cl_mem), &deviceData->m_dGlobsum), "Could not set global sum parameter");
        V_RETURN_CL(clSetKernelArg(scanHistogramKernel, 2, sizeof(cl_mem), &deviceData->m_dTemp), "Could not set temporary parameter");
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
        0, NULL, &eve), "Could not execute 2nd instance of scanHistogram kernel.");

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

    scan_time += (float)(fin - debut) / 1e9f;
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
	size_t nblocitems = Parameters::_NUM_ITEMS_PER_GROUP;
    size_t nbitems    = Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS;

	assert(nkeys_rounded % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);

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

	// CONSIDER: Using std::tuple<cl_mem, cl_mem, ...>
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
        V_RETURN_CL(clSetKernelArg(reorderKernel, 0, sizeof(cl_mem), &deviceData->m_dInKeys), "Could not set input keys for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, 1, sizeof(cl_mem), &deviceData->m_dOutKeys), "Could not set output keys for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, 2, sizeof(cl_mem), &deviceData->m_dHistograms), "Could not set histograms for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, 3, sizeof(pass),   &pass), "Could not set pass for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, 4, sizeof(cl_mem), &deviceData->m_dInPermut), "Could not set input permutation for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, 5, sizeof(cl_mem), &deviceData->m_dOutPermut), "Could not set output permutation for reorder kernel.");
		V_RETURN_CL(clSetKernelArg(reorderKernel, 6,
			sizeof(cl_int) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP,
            NULL), "Could not set local memory for reorder kernel."); // mem cache

        V_RETURN_CL(clSetKernelArg(reorderKernel, 7, sizeof(nkeys_rounded), &nkeys_rounded), "Could not set number of input keys for reorder kernel.");
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
    cl_mem d_temp;
	d_temp	              = deviceData->m_dInKeys;
	deviceData->m_dInKeys  = deviceData->m_dOutKeys;
    deviceData->m_dOutKeys = d_temp;

    // swap the old and new permutations
    d_temp       = deviceData->m_dInPermut;
    deviceData->m_dInPermut  = deviceData->m_dOutPermut;
    deviceData->m_dOutPermut = d_temp;
}

/// Check divisibility of works to assign correct amounts of work to groups/work-items.
template <typename DataType>
void CRadixSortTask<DataType>::CheckDivisibility() 
{
    assert(Parameters::_RADIX == pow(2, Parameters::_NUM_BITS_PER_RADIX));
    assert(Parameters::_TOTALBITS % Parameters::_NUM_BITS_PER_RADIX == 0);
    assert(Parameters::_NUM_MAX_INPUT_ELEMS % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);
    assert((Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_RADIX) % Parameters::_NUM_HISTOSPLIT == 0);
    assert(pow(2, (int)log2(Parameters::_NUM_GROUPS)) == Parameters::_NUM_GROUPS);
    assert(pow(2, (int)log2(Parameters::_NUM_ITEMS_PER_GROUP)) == Parameters::_NUM_ITEMS_PER_GROUP);
}

template <typename DataType>
void CRadixSortTask<DataType>::CheckLocalMemory(cl_device_id Device) 
{
    // check that the local mem is sufficient (suggestion of Jose Luis Cerc\F3s Pita)
    cl_ulong localMem;
	clGetDeviceInfo(Device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, NULL);
    if (options.verbose) {
        cout << "Cache size   = " << localMem << " Bytes." << endl;
		cout << "Needed cache = " << sizeof(cl_uint) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP << " Bytes." << endl;
    }
	assert(localMem > sizeof(DataType) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP);

	unsigned int maxmemcache = max(Parameters::_NUM_HISTOSPLIT, 
		Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS * Parameters::_RADIX / Parameters::_NUM_HISTOSPLIT);
	assert(localMem > sizeof(DataType)*maxmemcache);
}

/// resize the sorted vector
template <typename DataType>
void CRadixSortTask<DataType>::Resize(int nn) 
{
	assert(nn <= Parameters::_NUM_MAX_INPUT_ELEMS);

    if (options.verbose){
        cout << "Resize to  " << nn << endl;
    }
    nkeys = nn;

    // length of the vector has to be divisible by (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP)
    int rest = nkeys % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP);
    nkeys_rounded = nkeys;

    if (rest != 0) {
		nkeys_rounded = nkeys - rest + (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP);
    }
	nkeys_rest = rest;
}

template <typename DataType>
void CRadixSortTask<DataType>::padGPUData(cl_command_queue CommandQueue)
{
	if (nkeys_rest != 0) {
		const auto MAX_INT = std::numeric_limits<DataType>::max();
		std::vector<DataType> pad(Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP, MAX_INT - 1);

		// pad the vector with big values
		assert(nkeys_rounded <= Parameters::_NUM_MAX_INPUT_ELEMS);

		const auto blocking = CL_TRUE;
		const auto offset = sizeof(DataType) * nkeys;
		const auto size = sizeof(DataType) * (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP - nkeys_rest);
		V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue,
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
void CRadixSortTask<DataType>::RadixSort(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
    CheckDivisibility();

	assert(nkeys_rounded <= Parameters::_NUM_MAX_INPUT_ELEMS);
    assert(nkeys <= nkeys_rounded);
	int nbcol = nkeys_rounded / (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP);
	int nbrow = Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP;

    if (options.verbose) {
        cout << "Start sorting " << nkeys << " keys." << endl;
    }

    for (int pass = 0; pass < Parameters::_NUM_PASSES; pass++){
        if (options.verbose) {
            cout << "Pass " << pass << ":" << endl;
        }

        if (options.verbose) {
            cout << "Building histograms" << endl;
        }
        Histogram(CommandQueue, pass);

        if (options.verbose) {
            cout << "Scanning histograms" << endl;
        }
        ScanHistogram(CommandQueue, pass);

        if (options.verbose) {
            cout << "Reordering " << endl;
        }
        Reorder(CommandQueue, pass);

        if (options.verbose) {
            cout << "-------------------" << endl;
        }
    }
    
    sort_time.avg = histo_time.avg + scan_time.avg + reorder_time.avg + paste_time.avg;
    sort_time.n = histo_time.n;
    if (options.verbose){
        cout << "End sorting" << endl;
    }
}

template <typename DataType>
void CRadixSortTask<DataType>::AllocateDeviceMemory(cl_context Context) 
{
	// Done in constructor of ComputeDeviceData :)
	Resize(nkeys);
	deviceData = std::make_shared<ComputeDeviceData<DataType>>(Context, nkeys_rounded);
}

template <typename DataType>
void CRadixSortTask<DataType>::CopyDataToDevice(cl_command_queue CommandQueue)
{
	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue,
        deviceData->m_dInKeys,
        CL_TRUE, 0,
        sizeof(DataType) * nkeys_rounded,
        hostData.m_hKeys.data(),
        0, NULL, NULL),
		"Could not initialize input keys device buffer");

    clFinish(CommandQueue);  // wait end of read

	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue,
        deviceData->m_dInPermut,
        CL_TRUE, 0,
        sizeof(uint32_t) * nkeys_rounded,
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
		sizeof(DataType) * nkeys_rounded,
        hostData.m_hResultGPUMap["RadixSort_01"].data(),
		0, NULL, NULL),
		"Could not read result data");

	clFinish(CommandQueue);  // wait end of read

	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue,
        deviceData->m_dInPermut,
		CL_TRUE, 0,
		sizeof(uint32_t)  * nkeys_rounded,
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
void CRadixSortTask<DataType>::ExecuteTask(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], const string& alternative)
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
bool CRadixSortTask<DataType>::writePerformanceToFile(const std::string& filename) 
{
    bool file_exists = false;
    {
        struct stat buffer;
        file_exists = (stat(filename.c_str(), &buffer) == 0);
    }

    std::ofstream outfile(filename, std::ofstream::out | std::ofstream::app);
    const std::vector<std::string> columns = {
        "NumElements", "Datatype", "Dataset", "avgHistogram", "avgScan", "avgPaste", "avgReorder", "avgTotalGPU", "avgTotalSTLCPU", "avgTotalRDXCPU"
    };
    // Print columns
    if (!file_exists)
    {
        outfile << columns[0];
        for (int i = 1; i < columns.size(); i++) {
            outfile << "," << columns[i];
        }
    }
    outfile << std::endl;
    outfile << nkeys << ",";
    outfile << TypeNameString<DataType>::stdint_name << ",";
    outfile << hostData.m_selectedDataset->getName() << ",";
    outfile << histo_time.avg << ",";
    outfile << scan_time.avg << ",";
    outfile << paste_time.avg << ",";
    outfile << reorder_time.avg << ",";
    outfile << sort_time.avg << ",";
    outfile << cpu_stl_time.avg << ",";
    outfile << cpu_radix_time.avg;
    outfile << std::endl;
    
    return true;
}

template <typename DataType>
void CRadixSortTask<DataType>::writePerformanceToStdout() 
{
    const std::vector<std::string> columns = {
        "NumElements", "Datatype", "Dataset", "avgHistogram", "avgScan", "avgPaste", "avgReorder", "avgTotalGPU", "avgTotalSTLCPU", "avgTotalRDXCPU"
    };
    // Print columns

    std::cout << columns[0];
    for (int i = 1; i < columns.size(); i++) {
        std::cout << "," << columns[i];
    }
    std::cout << std::endl;
    std::cout << nkeys << ",";
    std::cout << TypeNameString<DataType>::stdint_name << ",";
    std::cout << hostData.m_selectedDataset->getName() << ",";
    std::cout << histo_time.avg << ",";
    std::cout << scan_time.avg << ",";
    std::cout << paste_time.avg << ",";
    std::cout << reorder_time.avg << ",";
    std::cout << sort_time.avg << ",";
    std::cout << cpu_stl_time.avg << ",";
    std::cout << cpu_radix_time.avg;
    std::cout << std::endl;
}

template <typename DataType>
void CRadixSortTask<DataType>::TestPerformance(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int Task)
{
    if (options.perf_to_stdout) {
        cout << " radixsort cpu avg time: " << cpu_radix_time.avg << " ms, throughput: " << 1.0e-6 * (double)nkeys_rounded / cpu_radix_time.avg << " Gelem/s" << endl;
        cout << " stl cpu avg time: " << cpu_stl_time.avg << " ms, throughput: " << 1.0e-6 * (double)nkeys_rounded / cpu_stl_time.avg << " Gelem/s" << endl;
    }

    if (options.perf_to_stdout) {
        cout << "Testing performance of GPU task " << deviceData->kernelNames[Task] << endl;
    }

    //finish all before we start measuring the time
    V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");

    CTimer timer;
    timer.Start();

    //run the kernel N times
	for (auto i = 0; i < Parameters::_NUM_PERFORMANCE_ITERATIONS; i++) {
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

        cout << " kernel |    avg      |     min     |    max " << endl;
        cout << " -----------------------------------------------" << endl;
        cout << "  histogram: " << std::setw(8) << histo_time.avg << " | " << histo_time.min << " | " << histo_time.max << endl;
        cout << "  scan:      " << std::setw(8) << scan_time.avg << " | " << scan_time.min << " | " << scan_time.max << endl;
        cout << "  paste:     " << std::setw(8) << paste_time.avg << " | " << paste_time.min << " | " << paste_time.max << endl;
        cout << "  reorder:   " << std::setw(8) << reorder_time.avg << " | " << reorder_time.min << " | " << reorder_time.max << endl;
        cout << " -----------------------------------------------" << endl;
        cout << "  total:     " << ms << " ms, throughput: " << 1.0e-6 * (double)nkeys_rounded / ms << " Gelem/s" << endl;
    }
    
    using std::chrono::system_clock;
    std::time_t tt = system_clock::to_time_t(system_clock::now());
    struct std::tm * ptm = std::localtime(&tt);
    const std::string dateFormat = "%H-%M-%S";
    std::stringstream fileNameBuilder;

    fileNameBuilder << "radix_" << std::put_time(ptm, dateFormat.c_str()) << ".csv";

    if (options.perf_to_csv) {
        writePerformanceToFile(fileNameBuilder.str());
    }
    if (options.perf_csv_to_stdout) {
        writePerformanceToStdout();
    }
}

// Specialize CRadixSortTask for exactly these four types.
template class CRadixSortTask < int32_t >;
template class CRadixSortTask < int64_t >;
template class CRadixSortTask < uint32_t >;
template class CRadixSortTask < uint64_t >;
