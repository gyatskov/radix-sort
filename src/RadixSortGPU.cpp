#include "RadixSortGPU.h"

#include "ComputeDeviceData.h"

#include "Common/CTimer.h"
#include "Common/CLTypeInformation.h"
#include "Common/Util.hpp"
#include <CL/Utils/Utils.hpp>

#include <sstream>
#include <ranges>
#include <cassert>
#include <cmath>
#include <fstream>
#include <filesystem>

template<typename DataType>
void RadixSortGPU<DataType>::Histogram(cl::CommandQueue CommandQueue, int pass)
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

template <typename DataType>
void RadixSortGPU<DataType>::ScanHistogram(cl::CommandQueue CommandQueue)
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

template <typename DataType>
void RadixSortGPU<DataType>::Reorder(cl::CommandQueue CommandQueue, int pass)
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
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputKeys],
        &pattern,
        paddingOffset,
        size_bytes
    );
}

template <typename DataType>
uint32_t RadixSortGPU<DataType>::Resize(uint32_t nn) noexcept
{
    // length of the vector has to be divisible by (AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP)
    const int32_t rest = nn % AlgorithmConfiguration::_NUM_ITEMS;

    const int32_t delta = (rest != 0) * (- rest + AlgorithmConfiguration::_NUM_ITEMS);
    return nn + delta;
}

template <typename DataType>
OperationStatus RadixSortGPU<DataType>::uploadData(
    cl::CommandQueue CommandQueue
)
{
    CopyDataToDevice(CommandQueue);
    const auto error = CommandQueue.finish();  // wait until end of write
    using S = OperationStatus;
    return error == CL_SUCCESS ? S::OK : S::DATA_UPLOAD_FAILED;
}

template <typename DataType>
OperationStatus RadixSortGPU<DataType>::calculate(
    cl::CommandQueue CommandQueue
)
{
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

    return OperationStatus::OK;
}

template <typename DataType>
OperationStatus RadixSortGPU<DataType>::downloadData(
    cl::CommandQueue CommandQueue
)
{
    CopyDataFromDevice(CommandQueue);
    const auto error = CommandQueue.finish();
    using S = OperationStatus;
    return error == CL_SUCCESS ? S::OK : S::DATA_DOWNLOAD_FAILED;
}

template <typename DataType>
OperationStatus RadixSortGPU<DataType>::downloadKeys(
    cl::CommandQueue CommandQueue
)
{
    constexpr auto isBlocking = CL_FALSE;
    constexpr auto offset = 0U;
    auto error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputKeys],
        isBlocking,
        offset,
        sizeof(DataType) * mNumberKeysRounded,
        mHostSpans.m_hResultFromGPU.data()
    );
    if (error != CL_SUCCESS) return OperationStatus::DATA_DOWNLOAD_FAILED;
    error = CommandQueue.finish();
    using S = OperationStatus;
    return error == CL_SUCCESS ? S::OK : S::DATA_DOWNLOAD_FAILED;
}

template <typename DataType>
OperationStatus RadixSortGPU<DataType>::downloadIntermediate(
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
        mHostSpans.m_hHistograms.data()
    );
    if (error != CL_SUCCESS) return S::DATA_DOWNLOAD_FAILED;

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::Globsum],
        isBlocking, offset,
        sizeof(uint32_t) * AlgorithmConfiguration::_NUM_HISTOSPLIT,
        mHostSpans.m_hGlobsum.data()
    );
    if (error != CL_SUCCESS) return S::DATA_DOWNLOAD_FAILED;

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputPermutations],
        isBlocking, offset,
        sizeof(uint32_t) * mNumberKeysRounded,
        mHostSpans.h_Permut.data()
    );
    if (error != CL_SUCCESS) return S::DATA_DOWNLOAD_FAILED;

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::OutputPermutations],
        isBlocking, offset,
        sizeof(uint32_t) * mNumberKeysRounded,
        mHostSpans.h_OutputPermut.data()
    );
    if (error != CL_SUCCESS) return S::DATA_DOWNLOAD_FAILED;

    error = CommandQueue.finish();
    return error == CL_SUCCESS ? S::OK : S::DATA_DOWNLOAD_FAILED;
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
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputKeys],
        isBlocking,
        0,
        sizeof(DataType) * mNumberKeysRounded,
        mHostSpans.m_hKeys.data()
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueWriteBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputPermutations],
        isBlocking,
        0,
        sizeof(uint32_t) * mNumberKeysRounded,
        mHostSpans.h_Permut.data()
    );
    assert(error == CL_SUCCESS);
}

template <typename DataType>
void RadixSortGPU<DataType>::CopyDataFromDevice(cl::CommandQueue CommandQueue)
{
    constexpr auto isBlocking = CL_FALSE;
    constexpr auto offset = 0U;
    auto error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputKeys],
		isBlocking,
        offset,
		sizeof(DataType) * mNumberKeysRounded,
        mHostSpans.m_hResultFromGPU.data()
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputPermutations],
		isBlocking,
        offset,
		sizeof(uint32_t) * mNumberKeysRounded,
        mHostSpans.h_Permut.data()
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::Histograms],
		isBlocking,
        offset,
		sizeof(uint32_t) * AlgorithmConfiguration::_RADIX * AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP,
        mHostSpans.m_hHistograms.data()
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::Globsum],
		isBlocking,
        offset,
		sizeof(uint32_t)  * AlgorithmConfiguration::_NUM_HISTOSPLIT,
		mHostSpans.m_hGlobsum.data()
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

// ---------------------------------------------------------------------------
// Returns the short type suffix used in the pre-compiled kernel file names,
// e.g. "int32" for int32_t, "uint64" for uint64_t.
// ---------------------------------------------------------------------------
template <typename DataType>
std::string_view RadixSortGPU<DataType>::KernelTypeSuffix() noexcept
{
    return TypeNameString<DataType>::stdint_name;
}

// ---------------------------------------------------------------------------
// Returns true when the given device advertises support for loading SPIR-V IL
// programs via clCreateProgramWithIL / cl_khr_il_program / cl_khr_spir.
// ---------------------------------------------------------------------------
template <typename DataType>
bool RadixSortGPU<DataType>::DeviceSupportsSPIRV(const cl::Device& device) noexcept
{
    // OpenCL 2.1+ exposes CL_DEVICE_IL_VERSION as a non-empty string when IL
    // (SPIR-V) is supported.
    try {
        // CL_DEVICE_IL_VERSION is defined in CL/cl.h for OpenCL >= 2.1
        // cl.hpp / opencl.hpp wraps it as Device::getInfo<CL_DEVICE_IL_VERSION>()
        // but the constant may not be available when targeting CL 1.2 headers.
        // We fall back to extension string inspection in that case.
#if CL_TARGET_OPENCL_VERSION >= 210
        const std::string ilVersion = device.getInfo<CL_DEVICE_IL_VERSION>();
        if (!ilVersion.empty() && ilVersion != "")
            return true;
#endif
    } catch (...) {}

    // Extension-string check works for any OpenCL version
    try {
        const std::string exts = device.getInfo<CL_DEVICE_EXTENSIONS>();
        if (exts.find("cl_khr_il_program") != std::string::npos)
            return true;
        if (exts.find("cl_khr_spir") != std::string::npos)
            return true;
    } catch (...) {}

    return false;
}

// ---------------------------------------------------------------------------
// Reads a binary file from disk into a byte vector.
// Returns an empty vector on failure.
// ---------------------------------------------------------------------------
static std::vector<unsigned char> ReadBinaryFile(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        return {};
    const auto size = static_cast<std::streamsize>(f.tellg());
    if (size <= 0)
        return {};
    f.seekg(0, std::ios::beg);
    std::vector<unsigned char> buf(static_cast<std::size_t>(size));
    if (!f.read(reinterpret_cast<char*>(buf.data()), size))
        return {};
    return buf;
}

// ---------------------------------------------------------------------------
// Searches a list of directories for a kernel binary file named
//   RadixSort_<suffix>.<ext>
// Returns the first one found, or an empty path.
// ---------------------------------------------------------------------------
static std::filesystem::path FindKernelBinary(
    const std::string_view typeSuffix,
    const std::string_view ext)
{
    // Build the filename
    const std::string filename =
        "RadixSort_" + std::string(typeSuffix) + "." + std::string(ext);

    // Candidate directories, in order of preference:
    //   1. Directory injected by CMake at compile time (build-tree or install-tree)
    //   2. Relative paths from the current working directory
    //   3. Relative paths from the executable directory
    std::vector<std::filesystem::path> searchDirs;

#ifdef RADIXSORTCL_KERNEL_DIR
    searchDirs.emplace_back(RADIXSORTCL_KERNEL_DIR);
#endif
    searchDirs.emplace_back("kernels");
    searchDirs.emplace_back(".");

    // Also try relative to the executable using the OpenCL utils helper
    try {
        const std::string exeDir = cl::util::executable_folder();
        searchDirs.emplace_back(std::filesystem::path(exeDir) / "kernels");
        searchDirs.emplace_back(std::filesystem::path(exeDir));
        // Also check share/radixsortcl/kernels relative to the exe's prefix
        searchDirs.emplace_back(
            std::filesystem::path(exeDir).parent_path() / "share" / "radixsortcl" / "kernels");
    } catch (...) {}

    for (const auto& dir : searchDirs) {
        const auto candidate = dir / filename;
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && !ec)
            return candidate;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Tries to build a cl::Program from a pre-compiled binary (SPIR-V or SPIR).
// Returns {program, OK} on success.
// Returns {empty, NO_SOURCE_FOUND} when no binary file exists at all.
// Returns {empty, error_status} when a file was found but could not be used.
// ---------------------------------------------------------------------------
template <typename DataType>
std::pair<cl::Program, OperationStatus>
RadixSortGPU<DataType>::TryLoadPrecompiledProgram(
    const cl::Device&  device,
    const cl::Context& context)
{
    using S = OperationStatus;
    const std::string_view suffix  = KernelTypeSuffix();
    const std::string      options = BuildOptions();

    // ---- 1. Try SPIR-V (.spv) when device supports it ----
    if (DeviceSupportsSPIRV(device)) {
        const auto spirvPath = FindKernelBinary(suffix, "spv");
        if (!spirvPath.empty()) {
            const auto bytes = ReadBinaryFile(spirvPath);
            if (bytes.empty())
                return { cl::Program{}, S::LOADING_SPIRV_FAILED };

            // clCreateProgramWithILKHR is an extension function: load its
            // address via the platform before calling it.
            cl::Platform platform;
            device.getInfo(CL_DEVICE_PLATFORM, &platform);

            using Fn = cl_program(CL_API_CALL*)(
                cl_context, const void*, size_t, cl_int*);
            auto fn = reinterpret_cast<Fn>(
                clGetExtensionFunctionAddressForPlatform(
                    platform(), "clCreateProgramWithILKHR"));
            if (!fn)
                return { cl::Program{}, S::PROGRAM_IL_CREATION_FAILED };

            cl_int err = CL_SUCCESS;
            const cl_program rawProg = fn(
                context(), bytes.data(), bytes.size(), &err);
            if (err != CL_SUCCESS || rawProg == nullptr)
                return { cl::Program{}, S::PROGRAM_IL_CREATION_FAILED };

            cl::Program prog(rawProg, /* retain = */ false);
            const cl_int buildErr = prog.build(device, options.c_str());
            if (buildErr != CL_SUCCESS || prog() == nullptr)
                return { cl::Program{}, S::PROGRAM_IL_CREATION_FAILED };
            return { std::move(prog), S::OK };
        }
    }

    // ---- 2. Try SPIR LLVM bitcode (.bc) via cl_khr_spir ----
    //
    // The .bc file is produced by:  clang -target spir64 -emit-llvm -c
    // and is loaded via clCreateProgramWithBinary (cl_khr_spir extension).
    {
        const auto bcPath = FindKernelBinary(suffix, "bc");
        if (!bcPath.empty()) {
            const auto bytes = ReadBinaryFile(bcPath);
            if (bytes.empty())
                return { cl::Program{}, S::LOADING_BINARY_FAILED };

            // clCreateProgramWithBinary takes parallel device/size/ptr arrays
            const cl_device_id   devId       = device();
            const size_t         binSize     = bytes.size();
            const unsigned char* binPtr      = bytes.data();
            cl_int               binStatus   = CL_SUCCESS;
            cl_int               err         = CL_SUCCESS;
            const cl_program rawProg = clCreateProgramWithBinary(
                context(),
                1,
                &devId,
                &binSize,
                &binPtr,
                &binStatus,
                &err);
            if (err != CL_SUCCESS || binStatus != CL_SUCCESS || rawProg == nullptr)
                return { cl::Program{}, S::PROGRAM_BINARY_CREATION_FAILED };

            cl::Program prog(rawProg, /* retain = */ false);
            const cl_int buildErr = prog.build(device, options.c_str());
            if (buildErr != CL_SUCCESS || prog() == nullptr)
                return { cl::Program{}, S::PROGRAM_BINARY_CREATION_FAILED };
            return { std::move(prog), S::OK };
        }
    }

    // No pre-compiled binary found — signal caller to fall back to source JIT
    return { cl::Program{}, S::NO_SOURCE_FOUND };
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
        // ----------------------------------------------------------------
        // 1. Try pre-compiled binary (SPIR-V or native) first.
        //    TryLoadPrecompiledProgram() returns NO_SOURCE_FOUND when no
        //    binary file is present at all, which is the expected case when
        //    the library was built without an offline OpenCL compiler.
        // ----------------------------------------------------------------
        {
            auto [prog, status] = TryLoadPrecompiledProgram(Device, Context);
            if (status == S::OK) {
                mDeviceData->m_Program = std::move(prog);
            } else if (status != S::NO_SOURCE_FOUND) {
                // A binary was found but could not be loaded/built — report it.
                return status;
            }
            // status == NO_SOURCE_FOUND => fall through to source compilation
        }

        // ----------------------------------------------------------------
        // 2. Fall back to runtime source compilation when no binary was
        //    found or produced at build time.
        // ----------------------------------------------------------------
        if (mDeviceData->m_Program() == nullptr) {
            const auto preamble = BuildPreamble();
            std::string programCode;
            const auto candidates = make_array<std::string>(
                "RadixSort.cl",
                "kernels/RadixSort.cl"
            );
            bool foundFile = false;
            for(const auto& path : candidates) {
                // Both methods could throw.
                try {
                    // First try working directory,
                    programCode = cl::util::read_text_file(path.c_str());
                    if(programCode.length()) {
                        foundFile = true;
                        break;
                    }
                } catch(const cl::util::Error&) {
                }

                try {
                    // then folder relative to executable
                    programCode = cl::util::read_exe_relative_text_file(path.c_str());
                    if(programCode.length()) {
                        foundFile = true;
                        break;
                    }
                } catch(const cl::util::Error&) {
                }
            }
            if(!foundFile)
            {
                return S::NO_SOURCE_FOUND;
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
        appendToOptions(options, "_ITEMS", AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP); // number of items in a group
        appendToOptions(options, "_GROUPS", AlgorithmConfiguration::_NUM_GROUPS); // the number of virtual processors is AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP * AlgorithmConfiguration::_NUM_GROUPS
        appendToOptions(options, "_HISTOSPLIT", AlgorithmConfiguration::_NUM_HISTOSPLIT); // number of splits of the histogram
        appendToOptions(options, "_TOTALBITS", Parameters::_TOTALBITS);  // number of bits for the integer in the list (max=32)
        appendToOptions(options, "_BITS", AlgorithmConfiguration::_NUM_BITS_PER_RADIX);  // number of bits in the radix
        // max size of the sorted vector
        // it has to be divisible by  AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP * AlgorithmConfiguration::_NUM_GROUPS
        // (for other sizes, pad the list with big values)
        appendToOptions(options, "_N", AlgorithmConfiguration::_NUM_MAX_INPUT_ELEMS);// maximal size of the list
        //#define PERMUT  // store the final permutation
        ////////////////////////////////////////////////////////

        // the following parameters are computed from the previous
        appendToOptions(options, "_RADIX", AlgorithmConfiguration::_RADIX);//  radix  = 2^_BITS
        appendToOptions(options, "_PASS", Parameters::_NUM_PASSES); // number of needed passes to sort the list
        appendToOptions(options, "_HISTOSIZE", AlgorithmConfiguration::_HISTOSIZE);// size of the histogram
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

template <typename DataType>
OperationStatus RadixSortGPU<DataType>::sort(
    cl::Device device,
    cl::Context context,
    cl::CommandQueue queue,
    std::span<const DataType> input,
    std::vector<DataType>& output
)
{
    const uint32_t numElements = static_cast<uint32_t>(input.size());
    const uint32_t numRounded  = Resize(numElements);

    // Allocate all working buffers
    std::vector<DataType>  hKeys(numRounded);
    std::vector<DataType>  hResult(numRounded);
    std::vector<uint32_t>  hHistograms(AlgorithmConfiguration::_RADIX * AlgorithmConfiguration::_NUM_ITEMS);
    std::vector<uint32_t>  hGlobsum(AlgorithmConfiguration::_NUM_HISTOSPLIT);
    std::vector<uint32_t>  hPermut(numRounded);
    std::vector<uint32_t>  hOutPermut(numRounded);

    std::copy_n(input.begin(), numElements, hKeys.begin());
    std::iota(hPermut.begin(), hPermut.end(), 0U);

    HostSpans<DataType> spans {
        { hKeys.data(),       hKeys.size()       },
        { hHistograms.data(), hHistograms.size()  },
        { hGlobsum.data(),    hGlobsum.size()     },
        { hPermut.data(),     hPermut.size()      },
        { hOutPermut.data(),  hOutPermut.size()   },
        { hResult.data(),     hResult.size()      },
    };

    auto status = initialize(device, context, numElements, spans);
    if (status != OperationStatus::OK) return status;

    if (numRounded != numElements) {
        padGPUData(queue, sizeof(DataType) * numElements);
    }

    status = uploadData(queue);
    if (status != OperationStatus::OK) { release(); return status; }

    status = calculate(queue);
    if (status != OperationStatus::OK) { release(); return status; }

    status = downloadData(queue);
    if (status != OperationStatus::OK) { release(); return status; }

    output.assign(hResult.begin(), hResult.begin() + numElements);
    release();
    return OperationStatus::OK;
}

// Specialize RadixSortGPU for the supported types.
template class RadixSortGPU < int32_t >;
template class RadixSortGPU < int64_t >;
template class RadixSortGPU < uint32_t >;
template class RadixSortGPU < uint64_t >;

