#include "RadixSortGPU.h"

#include "ComputeDeviceData.h"

#include "../Common/CLUtil.h"
#include "../Common/CTimer.h"
#include "../Common/CLTypeInformation.h"

#include <sstream>
#include <cassert>
#include <cmath>

template<typename DataType>
void RadixSortGPU<DataType>::Histogram(cl_command_queue CommandQueue, int pass)
{
    const size_t nbitems = Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS;
    const size_t nblocitems = Parameters::_NUM_ITEMS_PER_GROUP;

	assert(mNumberKeysRounded % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);
	assert(mNumberKeysRounded <= Parameters::_NUM_MAX_INPUT_ELEMS);

	const auto histogramKernelHandle = mDeviceData->m_kernelMap["histogram"];

	// Set kernel arguments
	{
        cl_uint argIdx = 0U;
		V_RETURN_CL(clSetKernelArg(histogramKernelHandle, argIdx++, sizeof(cl_mem), &mDeviceData->m_dMemoryMap["inputKeys"]), "Could not set input elements argument");
		V_RETURN_CL(clSetKernelArg(histogramKernelHandle, argIdx++, sizeof(cl_mem), &mDeviceData->m_dMemoryMap["histograms"]), "Could not set input histograms");
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
    mRuntimesGPU.timeHisto.update(timer.GetElapsedMilliseconds());

#ifdef MORE_PROFILING
    cl_ulong debut, fin;
    cl_int err{-1};
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

    mRuntimesGPU.timeHisto += (float)(fin - debut) / 1e9f;
#endif
}

template <typename DataType>
void RadixSortGPU<DataType>::ScanHistogram(cl_command_queue CommandQueue)
{
    const cl_uint workDimension = 1;
    size_t* globalWorkOffset = nullptr;

    {
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

        const auto scanHistogramKernel  = mDeviceData->m_kernelMap["scanhistograms"];
        // Set kernel arguments
        {
            cl_uint argIdx = 0U;
            V_RETURN_CL(clSetKernelArg(scanHistogramKernel, argIdx++, sizeof(cl_mem), &mDeviceData->m_dMemoryMap["histograms"]), "Could not set histogram argument");
            V_RETURN_CL(clSetKernelArg(scanHistogramKernel, argIdx++, sizeof(uint32_t) * maxmemcache, NULL), "Could not set histogram cache size"); // mem cache
            V_RETURN_CL(clSetKernelArg(scanHistogramKernel, argIdx++, sizeof(cl_mem), &mDeviceData->m_dMemoryMap["globsum"]), "Could not set global histogram argument");
        }
        cl_event eve;
        CTimer timer;
        timer.Start();
        V_RETURN_CL(clEnqueueNDRangeKernel(CommandQueue,
            scanHistogramKernel,
            workDimension,
            globalWorkOffset,
            &nbitems,
            &nblocitems,
            0, NULL, &eve
        ), "Could not execute 1st instance of scanHistogram kernel.");

        clFinish(CommandQueue);
        timer.Stop();
        mRuntimesGPU.timeScan.update(timer.GetElapsedMilliseconds());

#ifdef MORE_PROFILING
        cl_int err = CL_SUCCESS;
        cl_ulong debut{0};
        cl_ulong fin{0};

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

        mRuntimesGPU.timeScan += (float)(fin - debut) / 1e9f;
#endif

        // second scan for the globsum
        // Set kernel arguments
        {
            V_RETURN_CL(
                clSetKernelArg(scanHistogramKernel,
                0,
                sizeof(cl_mem),
                &mDeviceData->m_dMemoryMap["globsum"]),
                "Could not set global sum parameter"
            );
            V_RETURN_CL(
                clSetKernelArg(scanHistogramKernel,
                2,
                sizeof(cl_mem),
                &mDeviceData->m_dMemoryMap["temp"]),
                "Could not set temporary parameter"
            );
        }

        {
            // global work size
            const size_t nbitems    = Parameters::_NUM_HISTOSPLIT / 2;
            // local work size
            const size_t nblocitems = nbitems;

            CTimer timer;
            timer.Start();
            // Execute kernel for second scan (global)
            V_RETURN_CL(clEnqueueNDRangeKernel(CommandQueue,
                scanHistogramKernel,
                workDimension,
                globalWorkOffset,
                &nbitems,
                &nblocitems,
                0, NULL, &eve),
                "Could not execute 2nd instance of scanHistogram kernel."
            );

            clFinish(CommandQueue);
            timer.Stop();
            mRuntimesGPU.timeScan.update(timer.GetElapsedMilliseconds());


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

            mRuntimesGPU.timeScan += static_cast<float>(fin - debut) / 1e9f;
#endif
        }
    }

    {
        // loops again in order to paste together the local histograms
        // global
        size_t nbitems    = Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP / 2;
        // local work size
        size_t nblocitems = nbitems / Parameters::_NUM_HISTOSPLIT;

        const auto pasteHistogramKernel = mDeviceData->m_kernelMap["pastehistograms"];
        // Set kernel arguments
        {
            cl_uint argIdx = 0U;
            V_RETURN_CL(clSetKernelArg(pasteHistogramKernel,
                        argIdx++,
                        sizeof(cl_mem),
                        &mDeviceData->m_dMemoryMap["histograms"]),
                    "Could not set histograms argument");
            V_RETURN_CL(clSetKernelArg(pasteHistogramKernel,
                        argIdx++,
                        sizeof(cl_mem),
                        &mDeviceData->m_dMemoryMap["globsum"]),
                    "Could not set globsum argument");
        }

        // Execute paste histogram kernel
        cl_event eve;
        CTimer timer;
        timer.Start();
        V_RETURN_CL(clEnqueueNDRangeKernel(CommandQueue,
            pasteHistogramKernel,
            workDimension,
            globalWorkOffset,
            &nbitems,
            &nblocitems,
            0, NULL, &eve),
            "Could not execute paste histograms kernel"
        );

        clFinish(CommandQueue);
        timer.Stop();
        mRuntimesGPU.timePaste.update(timer.GetElapsedMilliseconds());

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

        mRuntimesGPU.timeScan += (float)(fin - debut) / 1e9f;
#endif
    }
}

template <typename DataType>
void RadixSortGPU<DataType>::Reorder(cl_command_queue CommandQueue, int pass)
{
	const size_t nblocitems = Parameters::_NUM_ITEMS_PER_GROUP;
    const size_t nbitems    = Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS;

	assert(mNumberKeysRounded % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);

    clFinish(CommandQueue);
    auto reorderKernel = mDeviceData->m_kernelMap["reorder"];

	//  const __global int* d_inKeys,
	//  __global int* d_outKeys,
	//	__global int* d_Histograms,
	//	const int pass,
	//	__global int* d_inPermut,
	//	__global int* d_outPermut,
	//	__local int* loc_histo,
	//	const int n

    // TODO: Use
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
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(cl_mem), &mDeviceData->m_dMemoryMap["inputKeys"]), "Could not set input keys for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(cl_mem), &mDeviceData->m_dMemoryMap["outputKeys"]), "Could not set output keys for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(cl_mem), &mDeviceData->m_dMemoryMap["histograms"]), "Could not set histograms for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(pass),   &pass), "Could not set pass for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(cl_mem), &mDeviceData->m_dMemoryMap["inputPermutations"]), "Could not set input permutation for reorder kernel.");
        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(cl_mem), &mDeviceData->m_dMemoryMap["outputPermutations"]), "Could not set output permutation for reorder kernel.");
		V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++,
			sizeof(cl_int) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP,
            NULL), "Could not set local memory for reorder kernel."); // mem cache

        V_RETURN_CL(clSetKernelArg(reorderKernel, argIdx++, sizeof(mNumberKeysRounded), &mNumberKeysRounded), "Could not set number of input keys for reorder kernel.");
	}

	assert(Parameters::_RADIX == pow(2, Parameters::_NUM_BITS_PER_RADIX));

    cl_event eve;

    constexpr cl_uint workDimension = 1;
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
    mRuntimesGPU.timeReorder.update(timer.GetElapsedMilliseconds());

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

    mRuntimesGPU.timeReorder += (float)(fin - debut) / 1e9f;
#endif

    // swap the old and new vectors of keys
    std::swap(mDeviceData->m_dMemoryMap["inputKeys"], mDeviceData->m_dMemoryMap["outputKeys"]);

    // swap the old and new permutations
    std::swap(mDeviceData->m_dMemoryMap["inputPermutations"], mDeviceData->m_dMemoryMap["outputPermutations"]);
}

template <typename DataType>
void RadixSortGPU<DataType>::padGPUData(
        cl_command_queue CommandQueue,
        size_t paddingOffset)
{
    constexpr auto MaxValue = std::numeric_limits<DataType>::max();
    // pads the vector with big values
    const auto pattern {MaxValue-1};
    const auto size_bytes = mNumberKeysRounded * sizeof(DataType) - paddingOffset;

    V_RETURN_CL(clEnqueueFillBuffer(
        CommandQueue,
        mDeviceData->m_dMemoryMap["inputKeys"],
        &pattern,
        sizeof(pattern),
        paddingOffset,
        size_bytes,
        0, NULL, NULL),
    "Could not pad input keys");
}

template <typename DataType>
uint32_t RadixSortGPU<DataType>::Resize(uint32_t nn) const noexcept
{
    // length of the vector has to be divisible by (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP)
    constexpr auto NumItems
        {Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP};
    const int32_t rest = nn % NumItems;

    const int32_t delta = (rest != 0) * (- rest + NumItems);
    return nn + delta;
}

template <typename DataType>
OperationStatus RadixSortGPU<DataType>::calculate(
    cl_command_queue CommandQueue
)
{
    CopyDataToDevice(CommandQueue);
    clFinish(CommandQueue);  // wait end of read

    for (uint32_t pass = 0U; pass < Parameters::_NUM_PASSES; pass++){
        if (mOutStream) {
            *mOutStream << "Pass " << pass << ":" << std::endl;
            *mOutStream << "Building histograms" << std::endl;
        }
        Histogram(CommandQueue, pass);

        if (mOutStream) {
            *mOutStream << "Scanning histograms" << std::endl;
        }
        ScanHistogram(CommandQueue);

        if (mOutStream) {
            *mOutStream << "Reordering " << std::endl;
        }
        Reorder(CommandQueue, pass);

        if (mOutStream) {
            *mOutStream << "-------------------" << std::endl;
        }
    }

    mRuntimesGPU.timeTotal.avg =
        mRuntimesGPU.timeHisto.avg
        + mRuntimesGPU.timeScan.avg
        + mRuntimesGPU.timeReorder.avg
        + mRuntimesGPU.timePaste.avg;

    mRuntimesGPU.timeTotal.n = mRuntimesGPU.timeHisto.n;
    CopyDataFromDevice(CommandQueue);
	clFinish(CommandQueue);  // wait end of read

    return OperationStatus::OK;
}

template <typename DataType>
void RadixSortGPU<DataType>::setLogStream(std::ostream* out) noexcept
{
    mOutStream = out;
}

template <typename DataType>
void RadixSortGPU<DataType>::CopyDataToDevice(cl_command_queue CommandQueue)
{
	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue,
        mDeviceData->m_dMemoryMap["inputKeys"],
        CL_TRUE, 0,
        sizeof(DataType) * mNumberKeysRounded,
        mHostSpans.m_hKeys.data,
        0, NULL, NULL),
		"Could not initialize input keys device buffer");

	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue,
        mDeviceData->m_dMemoryMap["inputPermutations"],
        CL_TRUE, 0,
        sizeof(uint32_t) * mNumberKeysRounded,
        mHostSpans.h_Permut.data,
        0, NULL, NULL),
		"Could not initialize input permutation device buffer");
}

template <typename DataType>
void RadixSortGPU<DataType>::CopyDataFromDevice(cl_command_queue CommandQueue)
{
	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue,
        mDeviceData->m_dMemoryMap["inputKeys"],
		CL_TRUE, 0,
		sizeof(DataType) * mNumberKeysRounded,
        mHostSpans.m_hResultFromGPU.data,
		0, NULL, NULL),
		"Could not read result data");

	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue,
        mDeviceData->m_dMemoryMap["inputPermutations"],
		CL_TRUE, 0,
		sizeof(uint32_t)  * mNumberKeysRounded,
        mHostSpans.h_Permut.data,
		0, NULL, NULL),
		"Could not read result permutation");

	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue,
        mDeviceData->m_dMemoryMap["histograms"],
		CL_TRUE, 0,
		sizeof(uint32_t)  * Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP,
        mHostSpans.m_hHistograms.data,
		0, NULL, NULL),
		"Could not read result histograms");

	V_RETURN_CL(clEnqueueReadBuffer(CommandQueue,
        mDeviceData->m_dMemoryMap["globsum"],
		CL_TRUE, 0,
		sizeof(uint32_t)  * Parameters::_NUM_HISTOSPLIT,
		mHostSpans.m_hGlobsum.data,
		0, NULL, NULL),
		"Could not read result global sum");
}

template <typename DataType>
std::string RadixSortGPU<DataType>::BuildPreamble()
{
    using UnsignedType = typename std::make_unsigned<DataType>::type;

    const auto OFFSET { -std::numeric_limits<DataType>::min() };
    std::stringstream ss;
    ss << "#define DataType " << TypeNameString<DataType>::open_cl_name << std::endl
       << "#define UnsignedDataType " << TypeNameString<UnsignedType>::open_cl_name << std::endl
       << "#define OFFSET " << OFFSET << std::endl;
    return ss.str();
}


template <typename DataType>
OperationStatus RadixSortGPU<DataType>::initialize(
    cl_device_id Device,
    cl_context Context,
    uint32_t nn,
    const HostSpans<DataType>& hostSpans
)
{
    using S = OperationStatus;

    // handle host buffers and init context
    {
        mNumberKeysRounded = Resize(nn);
        mHostSpans = hostSpans;
        mDeviceData =
            std::make_shared<ComputeDeviceData<DataType>>(
                    Context,
                    mNumberKeysRounded);
    }

    // compile and build program
    {
        const auto preamble = BuildPreamble();
        std::string programCode;
        if(!CLUtil::LoadProgramSourceToMemory("RadixSort.cl", programCode)) {
            return S::LOADING_SOURCE_FAILED;
        }
        const auto completeCode = preamble + programCode;
        const auto options { BuildOptions() };
        mDeviceData->m_Program =
            CLUtil::BuildCLProgramFromMemory(
                Device,
                Context,
                completeCode,
                options
            );
        if (mDeviceData->m_Program == nullptr) {
            return S::PROGRAM_CREATION_FAILED;
        }
    }

    // create individual kernels into just created program
    {
        cl_int clError{-1};
        for (const auto& kernelName : mDeviceData->kernelNames) {
            // Input data stays the same for each kernel
            mDeviceData->m_kernelMap[kernelName] =
                clCreateKernel(
                    mDeviceData->m_Program,
                    kernelName.c_str(),
                    &clError
            );

            // TODO: Use enum->str mapping for errors
            const auto errorMsg { std::string("Failed to create kernel: ") + kernelName };
            V_RETURN_CUSTOM_CL(clError, errorMsg.c_str(), S::KERNEL_CREATION_FAILED);
        }
    }
    return S::OK;
}

template <typename T>
typename std::enable_if_t<!std::is_integral<T>::value>
appendToOptions(std::string& dst, const std::string& key, const T& obj)
{
    dst += " -D" + key + "=" + "'" + std::string(obj) + "'";
}

template<typename T>
typename std::enable_if_t<std::is_integral<T>::value>
appendToOptions(std::string& dst, const std::string& key, const T& value)
{
    dst += " -D" + key + "=" + "'" + std::to_string(value) + "'";
}

template <typename DataType>
std::string RadixSortGPU<DataType>::BuildOptions()
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
RuntimesGPU RadixSortGPU<DataType>::getRuntimes() const
{
    return mRuntimesGPU;
}

// Specialize CRadixSortTask for the supported types.
template class RadixSortGPU < int32_t >;
template class RadixSortGPU < int64_t >;
template class RadixSortGPU < uint32_t >;
template class RadixSortGPU < uint64_t >;

