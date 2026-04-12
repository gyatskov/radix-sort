#include "RadixSortGPU.h"

#include "ComputeDeviceData.h"

#include "Common/CTimer.h"
#include "Common/Util.hpp"
#include "Parameters.h"

#include <cassert>
#include <cmath>

void RadixSortGPU::Histogram(cl::CommandQueue CommandQueue, int pass)
{
    const size_t nbitems = AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP * AlgorithmConfiguration::_NUM_GROUPS;
    const size_t nblocitems = AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP;

	assert(mNumberKeysRounded % (AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP) == 0);
	assert(mNumberKeysRounded <= AlgorithmConfiguration::_NUM_MAX_INPUT_ELEMS);

	auto histogramKernelHandle = mDeviceData->m_kernelMap["histogram"];

	// Set kernel arguments
	{
        const auto localCacheSize = sizeof(cl_int) * AlgorithmConfiguration::_RADIX * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP;
        cl_uint argIdx = 0U;
        histogramKernelHandle.setArg(argIdx++, mDeviceData->m_dMemoryMap[MemoryBuffer::InputKeys]);
        histogramKernelHandle.setArg(argIdx++, mDeviceData->m_dMemoryMap[MemoryBuffer::Histograms]);
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

void RadixSortGPU::ScanHistogram(cl::CommandQueue CommandQueue)
{
    {
        // numbers of processors for the local scan
        // = half the size of the local histograms
        // global work size
        size_t nbitems    = AlgorithmConfiguration::_RADIX * AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP / 2;
        // local work size
        size_t nblocitems = nbitems / AlgorithmConfiguration::_NUM_HISTOSPLIT;

        const uint32_t maxmemcache = std::max(AlgorithmConfiguration::_NUM_HISTOSPLIT,
            AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP * AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_RADIX / AlgorithmConfiguration::_NUM_HISTOSPLIT);

        // scan locally the histogram (the histogram is split into several
        // parts that fit into the local memory)

        auto scanHistogramKernel  = mDeviceData->m_kernelMap["scanhistograms"];
        // Set kernel arguments
        {
            cl_uint argIdx = 0U;

            scanHistogramKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap[MemoryBuffer::Histograms]);
            scanHistogramKernel.setArg(argIdx++, cl::Local(sizeof(uint32_t) * maxmemcache));
            scanHistogramKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap[MemoryBuffer::Globsum]);
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
            scanHistogramKernel.setArg(0,mDeviceData->m_dMemoryMap[MemoryBuffer::Globsum]);
            scanHistogramKernel.setArg(2,mDeviceData->m_dMemoryMap[MemoryBuffer::Temp]);
        }

        {
            // global work size
            const size_t nbitems    = AlgorithmConfiguration::_NUM_HISTOSPLIT / 2;
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
        size_t nbitems    = AlgorithmConfiguration::_RADIX * AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP / 2;
        // local work size
        size_t nblocitems = nbitems / AlgorithmConfiguration::_NUM_HISTOSPLIT;

        auto pasteHistogramKernel = mDeviceData->m_kernelMap["pastehistograms"];
        // Set kernel arguments
        {
            cl_uint argIdx = 0U;
            pasteHistogramKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap[MemoryBuffer::Histograms]);
            pasteHistogramKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap[MemoryBuffer::Globsum]);
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

void RadixSortGPU::Reorder(cl::CommandQueue CommandQueue, int pass)
{
	constexpr size_t nblocitems = AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP;
    constexpr size_t nbitems    = AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP * AlgorithmConfiguration::_NUM_GROUPS;

	assert(mNumberKeysRounded % (AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP) == 0);

    CommandQueue.finish();
    auto reorderKernel = mDeviceData->m_kernelMap["reorder"];
	assert(AlgorithmConfiguration::_RADIX == pow(2, AlgorithmConfiguration::_NUM_BITS_PER_RADIX));

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
        reorderKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap[MemoryBuffer::InputKeys]);
        reorderKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap[MemoryBuffer::OutputKeys]);
        reorderKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap[MemoryBuffer::Histograms]);
        reorderKernel.setArg(argIdx++, pass);
        reorderKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap[MemoryBuffer::InputPermutations]);
        reorderKernel.setArg(argIdx++, mDeviceData->m_dMemoryMap[MemoryBuffer::OutputPermutations]);
        reorderKernel.setArg(argIdx++, cl::Local(sizeof(cl_int) * AlgorithmConfiguration::_RADIX * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP));
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
    std::swap(mDeviceData->m_dMemoryMap[MemoryBuffer::InputKeys], mDeviceData->m_dMemoryMap[MemoryBuffer::OutputKeys]);

    // swap the old and new permutations
    std::swap(mDeviceData->m_dMemoryMap[MemoryBuffer::InputPermutations], mDeviceData->m_dMemoryMap[MemoryBuffer::OutputPermutations]);
}

uint32_t RadixSortGPU::Resize(uint32_t nn) noexcept
{
    // length of the vector has to be divisible by (AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP)
    const int32_t rest = nn % AlgorithmConfiguration::_NUM_ITEMS;

    const int32_t delta = (rest != 0) * (- rest + AlgorithmConfiguration::_NUM_ITEMS);
    return nn + delta;
}

OperationStatus RadixSortGPU::downloadIntermediate(
    cl::CommandQueue CommandQueue
)
{
    constexpr auto isBlocking = CL_FALSE;
    constexpr auto offset = 0U;
    using S = OperationStatus;

    auto error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::Histograms],
        isBlocking, offset,
        sizeof(uint32_t) * AlgorithmConfiguration::_RADIX * AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP,
        mHostSpans.hHistogramsData()
    );
    if (error != CL_SUCCESS) return S::DATA_DOWNLOAD_FAILED;

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::Globsum],
        isBlocking, offset,
        sizeof(uint32_t) * AlgorithmConfiguration::_NUM_HISTOSPLIT,
        mHostSpans.hGlobsumData()
    );
    if (error != CL_SUCCESS) return S::DATA_DOWNLOAD_FAILED;

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputPermutations],
        isBlocking, offset,
        sizeof(uint32_t) * mNumberKeysRounded,
        mHostSpans.hPermutData()
    );
    if (error != CL_SUCCESS) return S::DATA_DOWNLOAD_FAILED;

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::OutputPermutations],
        isBlocking, offset,
        sizeof(uint32_t) * mNumberKeysRounded,
        mHostSpans.hOutputPermutData()
    );
    if (error != CL_SUCCESS) return S::DATA_DOWNLOAD_FAILED;

    error = CommandQueue.finish();
    return error == CL_SUCCESS ? S::OK : S::DATA_DOWNLOAD_FAILED;
}

void RadixSortGPU::setLogStream(std::ostream* out) noexcept
{
    mOutStream = out;
}

OperationStatus RadixSortGPU::release()
{
    mDeviceData.reset();
    return OperationStatus::OK;
}

RuntimesGPU RadixSortGPU::getRuntimes() const
{
    return mRuntimesGPU;
}
