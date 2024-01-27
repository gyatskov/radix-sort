#include "RadixSortGPU.h"

#include "ComputeDeviceData.h"

#include "../Common/CTimer.h"
#include "../Common/CLTypeInformation.h"
#include "../Common/Util.hpp"
#include <CL/Utils/Utils.hpp>

#include <sstream>
#include <fstream>
#include <cassert>
#include <cmath>

template<typename DataType>
void RadixSortGPU<DataType>::Histogram(cl::CommandQueue CommandQueue, int pass)
{
    const size_t nbitems = Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS;
    const size_t nblocitems = Parameters::_NUM_ITEMS_PER_GROUP;

	assert(mNumberKeysRounded % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);
	assert(mNumberKeysRounded <= Parameters::_NUM_MAX_INPUT_ELEMS);

	auto histogramKernelHandle = mDeviceData->m_kernelMap["histogram"];

	// Set kernel arguments
	{
        const auto localCacheSize = sizeof(cl_int) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP;
        cl_uint argIdx = 0U;
        histogramKernelHandle.setArg(argIdx++, mDeviceData->m_dMemoryMap["inputKeys"]);
        histogramKernelHandle.setArg(argIdx++, mDeviceData->m_dMemoryMap["histograms"]);
        histogramKernelHandle.setArg(argIdx++, pass);
        histogramKernelHandle.setArg(argIdx++, cl::Local(localCacheSize));
        histogramKernelHandle.setArg(argIdx++, mNumberKeysRounded);
	}

    cl::Event event;
    CTimer timer;
    timer.Start();
    const cl::NDRange globalWorkOffset = cl::NullRange;
    const cl::NDRange globalWork{nbitems};
    const cl::NDRange localWork{nblocitems};
    const auto eventWaitList = nullptr;
	// Execute kernel
    const auto err = CommandQueue.enqueueNDRangeKernel(
            histogramKernelHandle,
            globalWorkOffset,
            globalWork,
            localWork,
            eventWaitList,
            &event
    );
    assert(err == CL_SUCCESS);
    CommandQueue.finish();
    timer.Stop();
    mRuntimesGPU.timeHisto.update(timer.GetElapsedMilliseconds());

#ifdef MORE_PROFILING
    mRuntimesGPU.timeHisto += cl::util::get_duration<CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_END>(event).count() / 1e9f;
#endif
}

template <typename DataType>
void RadixSortGPU<DataType>::ScanHistogram(cl::CommandQueue CommandQueue)
{
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

        auto scanHistogramKernel  = mDeviceData->m_kernelMap["scanhistograms"];
        // Set kernel arguments
        {
            cl_uint argIdx = 0U;

            scanHistogramKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap["histograms"]);
            scanHistogramKernel.setArg(argIdx++, cl::Local(sizeof(uint32_t) * maxmemcache));
            scanHistogramKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap["globsum"]);
        }
        cl::Event event;
        CTimer timer;
        timer.Start();
        const cl::NDRange globalWorkOffset = cl::NullRange;
        const cl::NDRange globalWork{nbitems};
        const cl::NDRange localWork{nblocitems};
        const auto eventWaitList = nullptr;
        const auto err = CommandQueue.enqueueNDRangeKernel(
             scanHistogramKernel,
             globalWorkOffset,
             globalWork,
             localWork,
             eventWaitList,
             &event
        );
        assert(err == CL_SUCCESS);

        CommandQueue.finish();
        timer.Stop();
        mRuntimesGPU.timeScan.update(timer.GetElapsedMilliseconds());

#ifdef MORE_PROFILING
        mRuntimesGPU.timeScan += cl::util::get_duration<CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_END>(event).count() / 1e9f;

#endif

        // second scan for the globsum
        // Set only first and third kernel arguments
        {
            scanHistogramKernel.setArg(0,mDeviceData->m_dMemoryMap["globsum"]);
            scanHistogramKernel.setArg(2,mDeviceData->m_dMemoryMap["temp"]);
        }

        {
            // global work size
            const size_t nbitems    = Parameters::_NUM_HISTOSPLIT / 2;
            // local work size
            const size_t nblocitems = nbitems;

            CTimer timer;
            timer.Start();
            const cl::NDRange globalWorkOffset = cl::NullRange;
            const cl::NDRange globalWork{nbitems};
            const cl::NDRange localWork{nblocitems};
            const auto eventWaitList = nullptr;
            // Execute kernel for second scan (global)
            const auto err = CommandQueue.enqueueNDRangeKernel(
                scanHistogramKernel,
                globalWorkOffset,
                globalWork,
                localWork,
                eventWaitList,
                &event
            );
            assert(err == CL_SUCCESS);

            CommandQueue.finish();
            timer.Stop();
            mRuntimesGPU.timeScan.update(timer.GetElapsedMilliseconds());

#ifdef MORE_PROFILING
            mRuntimesGPU.timeScan += cl::util::get_duration<CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_END>(event).count() / 1e9f;
#endif
        }
    }

    {
        // loops again in order to paste together the local histograms
        // global
        size_t nbitems    = Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP / 2;
        // local work size
        size_t nblocitems = nbitems / Parameters::_NUM_HISTOSPLIT;

        auto pasteHistogramKernel = mDeviceData->m_kernelMap["pastehistograms"];
        // Set kernel arguments
        {
            cl_uint argIdx = 0U;
            pasteHistogramKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap["histograms"]);
            pasteHistogramKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap["globsum"]);
        }

        // Execute paste histogram kernel
        cl::Event event;
        CTimer timer;
        timer.Start();
        const cl::NDRange globalWorkOffset = cl::NullRange;
        const cl::NDRange globalWork{nbitems};
        const cl::NDRange localWork{nblocitems};
        const auto eventWaitList = nullptr;
        const auto err = CommandQueue.enqueueNDRangeKernel(
            pasteHistogramKernel,
            globalWorkOffset,
            globalWork,
            localWork,
            eventWaitList,
            &event
        );
        assert(err == CL_SUCCESS);

        CommandQueue.finish();
        timer.Stop();
        mRuntimesGPU.timePaste.update(timer.GetElapsedMilliseconds());

#ifdef MORE_PROFILING
        mRuntimesGPU.timePaste += cl::util::get_duration<CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_END>(event).count() / 1e9f;
#endif
    }
}

template <typename DataType>
void RadixSortGPU<DataType>::Reorder(cl::CommandQueue CommandQueue, int pass)
{
	constexpr size_t nblocitems = Parameters::_NUM_ITEMS_PER_GROUP;
    constexpr size_t nbitems    = Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS;

	assert(mNumberKeysRounded % (Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP) == 0);

    CommandQueue.finish();
    auto reorderKernel = mDeviceData->m_kernelMap["reorder"];
	assert(Parameters::_RADIX == pow(2, Parameters::_NUM_BITS_PER_RADIX));

    // TODO: Use
	struct ReorderKernelParams {
        cl::Memory inKeys;
        cl::Memory outKeys;
        cl::Memory histograms;
		int pass;
        cl::Memory inPermutation;
        cl::Memory outPermutation;
		size_t localHistogramSize;
		int numElems;
	};

	// set kernel arguments
	{
        cl_uint argIdx = 0U;
        reorderKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap["inputKeys"]);
        reorderKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap["outputKeys"]);
        reorderKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap["histograms"]);
        reorderKernel.setArg(argIdx++, pass);
        reorderKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap["inputPermutations"]);
        reorderKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap["outputPermutations"]);
        reorderKernel.setArg(argIdx++, cl::Local(sizeof(cl_int) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP));
        reorderKernel.setArg(argIdx++, mNumberKeysRounded);
	}

    cl::Event event;

    const cl::NDRange globalWorkOffset = cl::NullRange;
    const cl::NDRange globalWork{nbitems};
    const cl::NDRange localWork{nblocitems};
    const auto eventWaitList = nullptr;
	// Execute kernel
    CTimer timer;
    timer.Start();
    const auto err = CommandQueue.enqueueNDRangeKernel(
		reorderKernel,
        globalWorkOffset,
        globalWork,
        localWork,
        eventWaitList,
        &event
    );
    assert(err == CL_SUCCESS);
    CommandQueue.finish();
    timer.Stop();
    mRuntimesGPU.timeReorder.update(timer.GetElapsedMilliseconds());

#ifdef MORE_PROFILING
    mRuntimesGPU.timeReorder += cl::util::get_duration<CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_END>(event).count() / 1e9f;

#endif

    // swap the old and new vectors of keys
    std::swap(mDeviceData->m_dMemoryMap["inputKeys"], mDeviceData->m_dMemoryMap["outputKeys"]);

    // swap the old and new permutations
    std::swap(mDeviceData->m_dMemoryMap["inputPermutations"], mDeviceData->m_dMemoryMap["outputPermutations"]);
}

template <typename DataType>
void RadixSortGPU<DataType>::padGPUData(
        cl::CommandQueue CommandQueue,
        size_t paddingOffset)
{
    constexpr auto MaxValue = std::numeric_limits<DataType>::max();
    // pads the vector with big values
    const auto pattern {MaxValue-1};
    const auto size_bytes = mNumberKeysRounded * sizeof(DataType) - paddingOffset;

    CommandQueue.enqueueFillBuffer(
        mDeviceData->m_dMemoryMap["inputKeys"],
        &pattern,
        paddingOffset,
        size_bytes
    );
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
    cl::CommandQueue CommandQueue
)
{
    CopyDataToDevice(CommandQueue);
    CommandQueue.finish();  // wait end of read

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
	CommandQueue.finish();  // wait until end of read

    return OperationStatus::OK;
}

template <typename DataType>
void RadixSortGPU<DataType>::setLogStream(std::ostream* out) noexcept
{
    mOutStream = out;
}

template <typename DataType>
void RadixSortGPU<DataType>::CopyDataToDevice( cl::CommandQueue CommandQueue)
{
    constexpr auto isBlocking = CL_FALSE;
    auto error = CL_SUCCESS;
    error = CommandQueue.enqueueWriteBuffer(
        mDeviceData->m_dMemoryMap["inputKeys"],
        isBlocking,
        0,
        sizeof(DataType) * mNumberKeysRounded,
        mHostSpans.m_hKeys.data
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueWriteBuffer(
        mDeviceData->m_dMemoryMap["inputPermutations"],
        isBlocking,
        0,
        sizeof(uint32_t) * mNumberKeysRounded,
        mHostSpans.h_Permut.data
    );
    assert(error == CL_SUCCESS);
}

template <typename DataType>
void RadixSortGPU<DataType>::CopyDataFromDevice(cl::CommandQueue CommandQueue)
{
    constexpr auto isBlocking = CL_FALSE;
    constexpr auto offset = 0U;
    auto error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap["inputKeys"],
		isBlocking,
        offset,
		sizeof(DataType) * mNumberKeysRounded,
        mHostSpans.m_hResultFromGPU.data
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap["inputPermutations"],
		isBlocking,
        offset,
		sizeof(uint32_t) * mNumberKeysRounded,
        mHostSpans.h_Permut.data
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap["histograms"],
		isBlocking,
        offset,
		sizeof(uint32_t) * Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP,
        mHostSpans.m_hHistograms.data
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap["globsum"],
		isBlocking,
        offset,
		sizeof(uint32_t)  * Parameters::_NUM_HISTOSPLIT,
		mHostSpans.m_hGlobsum.data
    );
    assert(error == CL_SUCCESS);
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
OperationStatus RadixSortGPU<DataType>::release()
{
    mDeviceData = nullptr;
    return OperationStatus::OK;
}

template <typename DataType>
OperationStatus RadixSortGPU<DataType>::initialize(
    cl::Device Device,
    cl::Context Context,
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
        std::string programCode = "";
        const auto checkedPaths = make_array<std::string>(
            "RadixSort.cl",
            "kernels/RadixSort.cl"
        );
        for(const auto& path : checkedPaths) {
            // Both methods could throw.
            try {
                // First try working directory,
                programCode = cl::util::read_text_file(path.c_str());
                if(programCode.length()) {
                    break;
                }
                // then folder relative to executable
                programCode = cl::util::read_exe_relative_text_file(path.c_str());
                if(programCode.length()) {
                    break;
                }
            } catch(const cl::util::Error& err) {
                continue;
            }
        }

        if(programCode.length() == 0)
        {
            return S::LOADING_SOURCE_FAILED;
        }
        const auto completeCode = preamble + programCode;

        const auto options { BuildOptions() };
        mDeviceData->m_Program = cl::Program(Context, completeCode);
        mDeviceData->m_Program.build(Device, options.c_str());

        if (mDeviceData->m_Program() == nullptr) {
            return S::PROGRAM_CREATION_FAILED;
        }
    }

    // create individual kernels into just created program
    {
        cl_int clError{-1};
        for (const auto& kernelName : mDeviceData->kernelNames) {
            // Input data stays the same for each kernel
            mDeviceData->m_kernelMap[kernelName] =
                cl::Kernel(
                    mDeviceData->m_Program,
                    kernelName.c_str(),
                    &clError
                );

            // TODO: Use enum->str mapping for errors
            const auto errorMsg { std::string("Failed to create kernel: ") + kernelName };
            if(clError) {
                std::cerr<<cl::util::Error(clError, errorMsg.c_str()).what()<<"\n";
                return S::KERNEL_CREATION_FAILED;
            }
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

