/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CRadixSortTask.h"

#include "../Common/CLUtil.h"
#include "../Common/CTimer.h"

#include <random>
#include <cassert>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CReductionTask

string g_kernelNames[] = {
    "RadixSort",
    "RadixSortReadWrite",

    "histogram",
    "scanhistograms",
    "pastehistograms",
    "reorder",

    "transpose"
};

CRadixSortTask::CRadixSortTask(size_t ArraySize)
	:
	m_hInput(ArraySize),
	m_resultCPU(ArraySize),
	m_dResultArray(),
	m_Program(NULL)
{
}

CRadixSortTask::~CRadixSortTask()
{
	ReleaseResources();
}

template<typename T>
void appendToOptions(std::string& dst, const std::string& key, const T& value) {
    dst += " -D" + key + "=" + to_string(value);
}

bool CRadixSortTask::InitResources(cl_device_id Device, cl_context Context)
{
	//CPU resources

	std::string seedStr("nico ist schmutz :)");
	std::seed_seq seed(seedStr.begin(), seedStr.end());
	std::mt19937 generator(seed);
	std::uniform_int_distribution<DataType> dis(0, std::numeric_limits<DataType>::max());
	//fill the array with some values
	for (size_t i = 0; i < m_N; i++) {
		//m_hInput[i] = m_N - i;			// Use this for debugging

		// Mersienne twister
		m_hInput[i] = dis(generator);

		//m_hInput[i] = rand() & 15;
	}

	//std::copy(m_hInput.begin(),
	//	m_hInput.begin() + 100,
	//	std::ostream_iterator<DataType>(std::cout, "\n"));

	// allocate device resources
	cl_int clError, clError2;
	m_dInKeys = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(DataType) * Parameters::_NUM_MAX_INPUT_ELEMS, NULL, &clError);
	m_dOutKeys = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(DataType) * Parameters::_NUM_MAX_INPUT_ELEMS, NULL, &clError);
	m_dInPermut = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(uint32_t)* Parameters::_NUM_MAX_INPUT_ELEMS, NULL, &clError);
	m_dOutPermut = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(uint32_t)* Parameters::_NUM_MAX_INPUT_ELEMS, NULL, &clError);

	V_RETURN_FALSE_CL(clError, "Error allocating device arrays");

	//load and compile kernels
	string programCode;
	size_t programSize = 0;

	//std::string pathToRadixSort("\\\\studhome.ira.uka.de\\s_yatsko\\windows\\folders\\Documents\\Visual Studio 2013\\Projects\\paralgo - radix - sort\\AssignmentRadixSort");

	CLUtil::LoadProgramSourceToMemory("RadixSort.cl", programCode);
    std::string options;
    //options += " -cl-opt-disable";

    // Compile options string
    {
        ///////////////////////////////////////////////////////
        // these parameters can be changed
        appendToOptions(options, "_ITEMS",		Parameters::_NUM_ITEMS_PER_GROUP); // number of items in a group
        appendToOptions(options, "_GROUPS",		Parameters::_NUM_GROUPS); // the number of virtual processors is Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS
        appendToOptions(options, "_HISTOSPLIT", Parameters::_NUM_HISTOSPLIT); // number of splits of the histogram
        appendToOptions(options, "_TOTALBITS",  Parameters::_TOTALBITS);  // number of bits for the integer in the list (max=32)
        appendToOptions(options, "_BITS",       Parameters::_NUM_BITS_PER_RADIX);  // number of bits in the radix
        // max size of the sorted vector
        // it has to be divisible by  Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS
        // (for other sizes, pad the list with big values)
        appendToOptions(options, "_N",          Parameters::_NUM_MAX_INPUT_ELEMS);// maximal size of the list

        if (Parameters::VERBOSE) {
            appendToOptions(options, "VERBOSE", Parameters::VERBOSE);
        }
        if (Parameters::TRANSPOSE) { 
            appendToOptions(options, "TRANSPOSE", Parameters::TRANSPOSE); // transpose the initial vector (faster memory access)
        }
        //#define PERMUT  // store the final permutation
        ////////////////////////////////////////////////////////

        // the following parameters are computed from the previous
        appendToOptions(options, "_RADIX",      Parameters::_RADIX);//  radix  = 2^_BITS
        appendToOptions(options, "_PASS",       Parameters::_NUM_PASSES); // number of needed passes to sort the list
        appendToOptions(options, "_HISTOSIZE",  Parameters::_HISTOSIZE);// size of the histogram
        // maximal value of integers for the sort to be correct
        appendToOptions(options, "_MAXINT",     Parameters::_MAXINT);
    }

	m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode, options);
    if (m_Program == nullptr) {
        return false;
    }

	// Create each kernel in global kernel list
    for (const auto& kernelName : g_kernelNames) {
        m_kernelMap[kernelName] = clCreateKernel(m_Program, kernelName.c_str(), &clError);
		m_resultGPUMap[kernelName] = std::vector<DataType>(m_N);
        std::string errorMsg("Failed to create kernel: ");
        errorMsg += kernelName;
        V_RETURN_FALSE_CL(clError, errorMsg.c_str());
    }

	return true;
}

void CRadixSortTask::ReleaseResources()
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

void CRadixSortTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	ExecuteTask(Context, CommandQueue, LocalWorkSize, "RadixSort");
	//ExecuteTask(Context, CommandQueue, LocalWorkSize, "RadixSortReadWrite");



	//std::copy(m_resultGPU.begin(),
	//	m_resultGPU.begin() + 100,
	//	std::ostream_iterator<DataType>(std::cout, "\n"));
	//
	//cout << "-----------" << endl;

	//TestPerformance(Context, CommandQueue, LocalWorkSize, 0);
    //TestPerformance(Context, CommandQueue, LocalWorkSize, 1);
}

////////////// RADIX SORT //////////////////////

// A function to do counting sort of arr[] according to
// the digit represented by exp.
template <typename ElemType>
void countSort(std::vector<ElemType>& arr, int exp)
{
	const auto n = arr.size();
	std::vector<ElemType> output(n, 0); // output array
	int i = 0;
	size_t count[10] = { 0 };

	// Store count of occurrences in count[]
	for (i = 0; i < n; i++) {
		count[(arr[i] / exp) % 10]++;
	}

	// Change count[i] so that count[i] now contains actual
	// position of this digit in output[]
	for (i = 1; i < 10; i++) {
		count[i] += count[i - 1];
	}

	// Build the output array
	for (i = n - 1; i >= 0; i--) {
		output[count[(arr[i] / exp) % 10] - 1] = arr[i];
		count[(arr[i] / exp) % 10]--;
	}

	// Copy the output array to arr[], so that arr[] now
	// contains sorted numbers according to current digit
	for (i = 0; i < n; i++) {
		arr[i] = output[i];
	}
}

// The main function to that sorts arr[] of size n using
// Radix Sort
template<typename ElemType>
void radixsort(std::vector<ElemType>& arr)
{
	// Find the maximum number to know number of digits
	const auto m = std::max_element(arr.begin(), arr.end());

	// Do counting sort for every digit. Note that instead
	// of passing digit number, exp is passed. exp is 10^i
	// where i is current digit number
	for (int exp = 1; *m / static_cast<ElemType>(exp) > 0; exp *= 10) {
		countSort(arr, exp);
	}
}

////////////// RADIX SORT //////////////////////


void CRadixSortTask::ComputeCPU()
{
	CTimer timer;
	timer.Start();

	const unsigned int NUM_ITERATIONS = 10;
    for (unsigned int j = 0; j < NUM_ITERATIONS; j++) {
        m_resultCPU = m_hInput;
        //// Use this only for really basic testing of the actually pointless kernel
        //for (auto& val : m_resultCPU) {
        //    val *= val;
        //}

        // Reference sorting (STL quicksort):
        //std::sort(m_resultCPU.begin(), m_resultCPU.end());

		// Reference sorting (radixsort):
		radixsort(m_resultCPU);
    }
	timer.Stop();

    double ms = timer.GetElapsedMilliseconds() / double(NUM_ITERATIONS);
	cout << "  average time: " << ms << " ms, throughput: " << 1.0e-6 * (double)m_N / ms << " Gelem/s" <<endl;
}

bool CRadixSortTask::ValidateResults()
{
	bool success = true;

	for (const auto& kernelName : g_kernelNames)
	{
#define RADIXSORT_CL_NOT_YET_IMPLEMENTED
#ifdef RADIXSORT_CL_NOT_YET_IMPLEMENTED
		std::sort(m_resultGPUMap[kernelName].begin(), m_resultGPUMap[kernelName].end());
#endif
		bool equalData = memcmp(m_resultGPUMap[kernelName].data(), m_resultCPU.data(), m_resultGPUMap[kernelName].size() * sizeof(DataType)) == 0;

		if (!equalData)
		{
            cout << "Validation of radixsort kernel " << kernelName << " failed.";
			success = false;
		}
	}

	return success;
}

void CRadixSortTask::Histogram(cl_command_queue CommandQueue, int pass) {
    cl_int err;

    size_t nblocitems = Parameters::_NUM_ITEMS_PER_GROUP;
    size_t nbitems = Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP;

	assert(nkeys_rounded % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);
	assert(nkeys_rounded <= Parameters::_NUM_MAX_INPUT_ELEMS);

	auto kernel = m_kernelMap["histogram"];

	// Set kernel arguments
	{
		err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_dInKeys);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &m_dHistograms);
		err |= clSetKernelArg(kernel, 2, sizeof(pass), &pass);
		err |= clSetKernelArg(kernel, 3, sizeof(Parameters::_RADIX) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP, NULL);
		err |= clSetKernelArg(kernel, 4, sizeof(int), &nkeys_rounded);
	}

    assert(err == CL_SUCCESS);

    cl_event eve;

	// Execute kernel
    err = clEnqueueNDRangeKernel(CommandQueue,
        kernel,
        1, NULL,
        &nbitems,
        &nblocitems,
        0, NULL, &eve);

    assert(err == CL_SUCCESS);

    clFinish(CommandQueue);

    cl_ulong debut, fin;

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

    histo_time += (float)(fin - debut) / 1e9;
}

void CRadixSortTask::ScanHistogram(cl_command_queue CommandQueue) {
    cl_int err;

    // numbers of processors for the local scan
    // = half the size of the local histograms
	size_t nbitems = Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP / 2;
	size_t nblocitems = nbitems / Parameters::_NUM_HISTOSPLIT;

	const uint32_t maxmemcache = max(Parameters::_NUM_HISTOSPLIT, 
		Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS * Parameters::_RADIX / Parameters::_NUM_HISTOSPLIT);

    // scan locally the histogram (the histogram is split into several
    // parts that fit into the local memory)

	auto scanHistogramKernel  = m_kernelMap["scanhistogram"];
	auto pasteHistogramKernel = m_kernelMap["pastehistogram"];

	err = clSetKernelArg(scanHistogramKernel, 0, sizeof(cl_mem), &m_dHistograms);
    assert(err == CL_SUCCESS);

	err = clSetKernelArg(scanHistogramKernel, 1,
        sizeof(uint32_t) * maxmemcache,
        NULL); // mem cache

	err = clSetKernelArg(scanHistogramKernel, 2, sizeof(cl_mem), &m_dGlobsum);
    assert(err == CL_SUCCESS);

    cl_event eve;

    err = clEnqueueNDRangeKernel(CommandQueue,
		scanHistogramKernel,
        1, NULL,
        &nbitems,
        &nblocitems,
        0, NULL, &eve);

    assert(err == CL_SUCCESS);
    clFinish(CommandQueue);

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

    scan_time += (float)(fin - debut) / 1e9;

    // second scan for the globsum
	err = clSetKernelArg(scanHistogramKernel, 0, sizeof(cl_mem), &m_dGlobsum);
    assert(err == CL_SUCCESS);

	err = clSetKernelArg(scanHistogramKernel, 2, sizeof(cl_mem), &m_dTemp);
    assert(err == CL_SUCCESS);

	nbitems = Parameters::_NUM_HISTOSPLIT / 2;
    nblocitems = nbitems;

    err = clEnqueueNDRangeKernel(CommandQueue,
		scanHistogramKernel,
        1, NULL,
        &nbitems,
        &nblocitems,
        0, NULL, &eve);

    assert(err == CL_SUCCESS);
    clFinish(CommandQueue);

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

    scan_time += (float)(fin - debut) / 1e9;

    // loops again in order to paste together the local histograms
	nbitems = Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP / 2;
	nblocitems = nbitems / Parameters::_NUM_HISTOSPLIT;

    err = clEnqueueNDRangeKernel(CommandQueue,
		pasteHistogramKernel,
        1, NULL,
        &nbitems,
        &nblocitems,
        0, NULL, &eve);

    assert(err == CL_SUCCESS);
    clFinish(CommandQueue);

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

    scan_time += (float)(fin - debut) / 1e9;
}

void CRadixSortTask::Reorder(cl_command_queue CommandQueue, int pass) {
    cl_int err;

	size_t nblocitems	= Parameters::_NUM_ITEMS_PER_GROUP;
	size_t nbitems		= Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP;

	assert(nkeys_rounded % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);

    clFinish(CommandQueue);
	auto reorderKernel = m_kernelMap["reorder"];

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
		err = clSetKernelArg(reorderKernel, 0, sizeof(cl_mem), &m_dInKeys);
		assert(err == CL_SUCCESS);

		err = clSetKernelArg(reorderKernel, 1, sizeof(cl_mem), &m_dOutKeys);
		assert(err == CL_SUCCESS);

		err = clSetKernelArg(reorderKernel, 2, sizeof(cl_mem), &m_dHistograms);
		assert(err == CL_SUCCESS);

		err = clSetKernelArg(reorderKernel, 3, sizeof(pass), &pass);
		assert(err == CL_SUCCESS);

		err = clSetKernelArg(reorderKernel, 4, sizeof(cl_mem), &m_dInPermut);
		assert(err == CL_SUCCESS);

		err = clSetKernelArg(reorderKernel, 5, sizeof(cl_mem), &m_dOutPermut);
		assert(err == CL_SUCCESS);

		err = clSetKernelArg(reorderKernel, 6,
			sizeof(uint32_t) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP,
			NULL); // mem cache
		assert(err == CL_SUCCESS);

		err = clSetKernelArg(reorderKernel, 7, sizeof(nkeys_rounded), &nkeys_rounded);
		assert(err == CL_SUCCESS);
	}

	assert(Parameters::_RADIX == pow(2, Parameters::_NUM_BITS_PER_RADIX));

    cl_event eve;

	// Execute kernel
    err = clEnqueueNDRangeKernel(CommandQueue,
		reorderKernel,
        1, NULL,
        &nbitems,
        &nblocitems,
        0, NULL, &eve);

    assert(err == CL_SUCCESS);
    clFinish(CommandQueue);

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

    reorder_time += (float)(fin - debut) / 1e9;

    // swap the old and new vectors of keys
    cl_mem d_temp;
	d_temp	 = m_dInKeys;
	m_dInKeys = m_dOutKeys;
    m_dOutKeys = d_temp;

    // swap the old and new permutations
    d_temp = m_dInPermut;
    m_dInPermut = m_dOutPermut;
    m_dOutPermut = d_temp;
}

// transpose the list for faster memory access
void CRadixSortTask::Transpose(int nbrow, int nbcol) {
#if 0 // not yet needed
    const int _TRANSBLOCK = 32; // size of the matrix block loaded into local memory
    int tilesize = _TRANSBLOCK;

    // if the matrix is too small, avoid using local memory
    if (nbrow%tilesize != 0) tilesize = 1;
    if (nbcol%tilesize != 0) tilesize = 1;

    if (tilesize == 1) {
        cout << "Warning, small list, avoiding cache..." << endl;
    }

    cl_int err;
    auto kernel = m_kernelMap["transpose"];
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_dInKeys);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &m_dOutKeys);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel, 2, sizeof(uint32_t), &nbcol);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel, 3, sizeof(uint32_t), &nbrow);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_inPermut);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_outPermut);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel, 6, sizeof(uint)*tilesize*tilesize, NULL);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel, 7, sizeof(uint)*tilesize*tilesize, NULL);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel, 8, sizeof(uint), &tilesize);
    assert(err == CL_SUCCESS);

    cl_event eve;

    size_t global_work_size[2];
    size_t local_work_size[2];

    assert(nbrow%tilesize == 0);
    assert(nbcol%tilesize == 0);

    global_work_size[0] = nbrow / tilesize;
    global_work_size[1] = nbcol;

    local_work_size[0] = 1;
    local_work_size[1] = tilesize;


    err = clEnqueueNDRangeKernel(CommandQueue,
        ckTranspose,
        2,   // two dimensions: rows and columns
        NULL,
        global_work_size,
        local_work_size,
        0, NULL, &eve);

    //exchange the pointers

    // swap the old and new vectors of keys
    cl_mem d_temp;
    d_temp = d_inKeys;
    d_inKeys = d_outKeys;
    d_outKeys = d_temp;

    // swap the old and new permutations
    d_temp = d_inPermut;
    d_inPermut = d_outPermut;
    d_outPermut = d_temp;


    // timing
    clFinish(CommandQueue);

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

    transpose_time += (float)(fin - debut) / 1e9;
#endif
}

/// Check divisibility of works to assign correct amounts of work to groups/work-items.
void CRadixSortTask::CheckDivisibility() {
    assert(Parameters::_RADIX == pow(2, Parameters::_NUM_BITS_PER_RADIX));
    assert(Parameters::_TOTALBITS % Parameters::_NUM_BITS_PER_RADIX == 0);
    assert(Parameters::_NUM_MAX_INPUT_ELEMS % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);
    assert((Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_RADIX) % Parameters::_NUM_HISTOSPLIT == 0);
    assert(pow(2, (int)log2(Parameters::_NUM_GROUPS)) == Parameters::_NUM_GROUPS);
    assert(pow(2, (int)log2(Parameters::_NUM_ITEMS_PER_GROUP)) == Parameters::_NUM_ITEMS_PER_GROUP);
}

void CRadixSortTask::CheckLocalMemory() {
#if 0
    // check that the local mem is sufficient (suggestion of Jose Luis Cercós Pita)
    cl_ulong localMem;
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, NULL);
    if (VERBOSE) {
        cout << "Cache size=" << localMem << " Bytes" << endl;
        cout << "Needed cache=" << sizeof(cl_uint)*_RADIX*Parameters::_NUM_ITEMS_PER_GROUP << " Bytes" << endl;
    }
    assert(localMem > sizeof(cl_uint)*_RADIX*Parameters::_NUM_ITEMS_PER_GROUP);

    unsigned int maxmemcache = max(_HISTOSPLIT, Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS * _RADIX / _HISTOSPLIT);
    assert(localMem > sizeof(cl_uint)*maxmemcache);
#endif
}

// resize the sorted vector
void CRadixSortTask::Resize(cl_command_queue CommandQueue, int nn) {
	assert(nn <= Parameters::_NUM_MAX_INPUT_ELEMS);

    if (Parameters::VERBOSE){
        cout << "Resize to  " << nn << endl;
    }
    nkeys = nn;

    // length of the vector has to be divisible by (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP)
    int reste = nkeys % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP);
    nkeys_rounded = nkeys;
    cl_int err;
    unsigned int pad[Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP];
    for (int ii = 0; ii < Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP; ii++){
        pad[ii] = Parameters::_MAXINT - (unsigned int)1;
    }
    if (reste != 0) {
        nkeys_rounded = nkeys - reste + (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP);
        // pad the vector with big values
		assert(nkeys_rounded <= Parameters::_NUM_MAX_INPUT_ELEMS);
        err = clEnqueueWriteBuffer(CommandQueue,
            m_dInKeys,
            CL_TRUE, sizeof(DataType) * nkeys,
			sizeof(DataType) *(Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP - reste),
            pad,
            0, NULL, NULL);
        //cout << nkeys<<" "<<nkeys_rounded<<endl;
        assert(err == CL_SUCCESS);
    }
}

void CRadixSortTask::RadixSort(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
    CheckDivisibility();
    CheckLocalMemory();

	assert(nkeys_rounded <= Parameters::_NUM_MAX_INPUT_ELEMS);
    assert(nkeys <= nkeys_rounded);
	int nbcol = nkeys_rounded / (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP);
	int nbrow = Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP;

    if (Parameters::VERBOSE) {
        cout << "Start sorting " << nkeys << " keys." << endl;
    }

#ifdef TRANSPOSE
    if (VERBOSE) {
        cout << "Transpose" << endl;
    }
    Transpose(nbrow, nbcol);
#endif

    for (int pass = 0; pass < Parameters::_NUM_PASSES; pass++){
        if (Parameters::VERBOSE) {
            cout << "Pass " << pass << ":" << endl;
        }

        if (Parameters::VERBOSE) {
            cout << "Building histograms" << endl;
        }
        Histogram(CommandQueue, pass);

        if (Parameters::VERBOSE) {
            cout << "Scanning histograms" << endl;
        }
        ScanHistogram(CommandQueue);

        if (Parameters::VERBOSE) {
            cout << "Reordering " << endl;
        }
        Reorder(CommandQueue, pass);

        if (Parameters::VERBOSE) {
            cout << "-------------------" << endl;
        }
    }
    
    if (Parameters::TRANSPOSE) {
        if (Parameters::VERBOSE) {
            cout << "Transposing" << endl;
        }
        Transpose(nbcol, nbrow);
    }
    
    //sort_time = histo_time + scan_time + reorder_time + transpose_time;
    if (Parameters::VERBOSE){
        cout << "End sorting" << endl;
    }
}

void CRadixSortTask::RadixSortReadWrite(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
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

void CRadixSortTask::CopyDataToDevice(cl_command_queue CommandQueue)
{
    cl_int status;

    status = clEnqueueWriteBuffer(CommandQueue,
        m_dInKeys,
        CL_TRUE, 0,
        sizeof(DataType) * Parameters::_NUM_MAX_INPUT_ELEMS,
        h_Keys,
        0, NULL, NULL);

    assert(status == CL_SUCCESS);
    clFinish(CommandQueue);  // wait end of read

    status = clEnqueueWriteBuffer(CommandQueue,
        m_dInPermut,
        CL_TRUE, 0,
        sizeof(uint32_t) * Parameters::_NUM_MAX_INPUT_ELEMS,
        h_Permut,
        0, NULL, NULL);

    assert(status == CL_SUCCESS);
    clFinish(CommandQueue);  // wait end of read
}

void CRadixSortTask::ExecuteTask(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], const string& alternative)
{
	//run selected task
	if (alternative == "RadixSort") {
        CopyDataToDevice(CommandQueue);
		RadixSort(Context, CommandQueue, LocalWorkSize);
	} else {
		V_RETURN_CL(false, "Invalid task selected");
	}

	//read back the results synchronously.
	std::fill(m_resultGPUMap[alternative].begin(), m_resultGPUMap[alternative].end(), 0);
}

void CRadixSortTask::TestPerformance(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int Task)
{
    cout << "Testing performance of task " << g_kernelNames[Task] << endl;

    //finish all before we start measuring the time
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
        }

    }

    //wait until the command queue is empty again
    V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");

    timer.Stop();

    double ms = timer.GetElapsedMilliseconds() / double(nIterations);
    cout << "  average time: " << ms << " ms, throughput: " << 1.0e-6 * (double)m_N / ms << " Gelem/s" << endl;
}

///////////////////////////////////////////////////////////////////////////////
