#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include "HostData.h"
#include "Statistics.h"
#include "OperationStatus.h"
#include "ComputeDeviceData.h"
#include "Common/CLTypeInformation.h"
#include "Common/Util.hpp"

#include <CL/Utils/Utils.hpp>

#include <memory>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cstdint>
#include <cassert>

/// Runtime statistics of GPU implementation algorithms
/// @note Radix sort specific
struct RuntimesGPU {
    Statistics timeHisto{};
    Statistics timeScan{};
    Statistics timeReorder{};
    Statistics timePaste{};
    Statistics timeTotal{};
};

namespace {
    
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
}

/// TODO: Avoid clFinish calls
///       For profiling use clGetEventProfilingInfo api
///       Provide more granular API:
///        +writeData()
///        +sort()
///        +readData()
class RadixSortGPU
{
public:
    /// 1. Creates program and kernel
    /// 2. Initializes host and device memory
  template <typename DataType>
  OperationStatus initialize(cl::Device Device, cl::Context Context,
                             uint32_t nn,
                             HostSpans<DataType> &hostSpans) {
    using S = OperationStatus;

    // handle host buffers and init context
    {
      mNumberKeysRounded = Resize(nn);
      mHostSpans = HostSpansProxy::FromHostSpans(hostSpans);
      mDeviceData =
          ComputeDeviceData::Create<DataType>(Context, mNumberKeysRounded);
    }

    // compile and build program
    {
      const auto preamble = BuildPreamble<DataType>();
      std::string programCode = "";
      const auto candidates = make_array<std::string>("RadixSort.cl", "kernels/RadixSort.cl");
      bool foundFile = false;
      for (const auto &path : candidates) {
        // Both methods could throw.
        try {
          // First try working directory,
          programCode = cl::util::read_text_file(path.c_str());
          if (programCode.length()) {
            foundFile = true;
            break;
          }
        } catch (const cl::util::Error &err) {
        }

        try {
          // then folder relative to executable
          programCode = cl::util::read_exe_relative_text_file(path.c_str());
          if (programCode.length()) {
            foundFile = true;
            break;
          }
        } catch (const cl::util::Error &err) {
        }
      }
      if (!foundFile) {
        return S::NO_SOURCE_FOUND;
      }

      if (programCode.empty()) {
        return S::LOADING_SOURCE_FAILED;
      }
      const auto completeCode = preamble + programCode;

      const auto options{BuildOptions<DataType>()};
      mDeviceData->m_Program = cl::Program(Context, completeCode);
      mDeviceData->m_Program.build(Device, options.c_str());

      if (mDeviceData->m_Program() == nullptr) {
        return S::PROGRAM_CREATION_FAILED;
      }
    }

    // create individual kernels into just created program
    {
      cl_int clError{-1};
      for (const auto &kernelName : mDeviceData->kernelNames) {
        // Input data stays the same for each kernel
        mDeviceData->m_kernelMap[kernelName] =
            cl::Kernel(mDeviceData->m_Program, kernelName.c_str(), &clError);

        // TODO: Use enum->str mapping for errors
        const auto errorMsg{std::string("Failed to create kernel: ") +
                            kernelName};
        if (clError) {
          std::cerr << cl::util::Error(clError, errorMsg.c_str()).what()
                    << "\n";
          return S::KERNEL_CREATION_FAILED;
        }
      }
    }
    return S::OK;
  }

    /// Copies host data to device
    /// @param CommandQueue OpenCL Command Queue
    template <typename DataType>
	OperationStatus uploadData(
        cl::CommandQueue CommandQueue
    ){
    CopyDataToDevice<DataType>(CommandQueue);
    const auto error = CommandQueue.finish();  // wait until end of write
    using S = OperationStatus;
    return error == CL_SUCCESS ? S::OK : S::DATA_UPLOAD_FAILED;
}

    /// Performs radix sort algorithm on previously provided data
    /// @param CommandQueue OpenCL Command Queue

    template <typename DataType>
	OperationStatus calculate(
        cl::CommandQueue CommandQueue
    ){
    for (uint32_t pass = 0U; pass < AlgorithmParameters<DataType>::_NUM_PASSES; pass++){
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

    /// Copies device data to host
    /// @param CommandQueue OpenCL Command Queue
    template <typename DataType>
	OperationStatus downloadData(
        cl::CommandQueue CommandQueue
    ){
    CopyDataFromDevice<DataType>(CommandQueue);
    const auto error = CommandQueue.finish();
    using S = OperationStatus;
    return error == CL_SUCCESS ? S::OK : S::DATA_DOWNLOAD_FAILED;
}

    /// Frees device buffers
    OperationStatus release();

    /// Sets output log stream
    /// @param[in,out] out Log text stream
    void setLogStream(std::ostream* out) noexcept;

    /// Rounds argument to next multiple of NumItems.
    /// @return Possibly rounded up number of elements
	static uint32_t Resize(uint32_t nn) noexcept;

    /// Pads GPU data buffers
    /// @param CommandQueue OpenCL Command Queue
    /// @param paddingOffset Padding offset in bytes
    template <typename DataType>
	void padGPUData(
        cl::CommandQueue CommandQueue,
        size_t paddingOffset
    ){
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

    /// Returns runtimes of individual algorithm steps
    /// @return runtimes of individual algorithm steps
    RuntimesGPU getRuntimes() const;

    /// One-call convenience method: allocates all internal buffers,
    /// initialises the GPU sorter, uploads, sorts, downloads, and cleans up.
    ///
    /// @param device   OpenCL device
    /// @param context  OpenCL context
    /// @param queue    OpenCL command queue
    /// @param input    Data to sort (read-only)
    /// @param[out] output  Will be resized and filled with the sorted result
    /// @return OperationStatus::OK on success
    template <typename DataType>
    OperationStatus sort(
        cl::Device device,
        cl::Context context,
        cl::CommandQueue queue,
        std::span<const DataType> input,
        std::vector<DataType>& output
    ){
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
        padGPUData<DataType>(queue, sizeof(DataType) * numElements);
    }

    status = uploadData<DataType>(queue);
    if (status != OperationStatus::OK) { release(); return status; }

    status = calculate<DataType>(queue);
    if (status != OperationStatus::OK) { release(); return status; }

    status = downloadData<DataType>(queue);
    if (status != OperationStatus::OK) { release(); return status; }

    output.assign(hResult.begin(), hResult.begin() + numElements);
    release();
    return OperationStatus::OK;
}

    /// @name Per-step methods for inspecting intermediate buffers
    /// Call these instead of calculate() to run one step at a time
    /// and download intermediate results between steps.
    /// @{

    /// Performs histogram calculation for a single pass
    void Histogram(cl::CommandQueue CommandQueue, int pass);
    /// Performs histogram scan
    void ScanHistogram(cl::CommandQueue CommandQueue);
    /// Performs reorder step for a single pass
    void Reorder(cl::CommandQueue CommandQueue, int pass);

    /// @}

    /// Downloads only the key buffer from device to host
    /// (writes into m_hResultFromGPU span provided at initialization).
    template <typename DataType>
    OperationStatus downloadKeys(cl::CommandQueue CommandQueue){
    constexpr auto isBlocking = CL_FALSE;
    constexpr auto offset = 0U;
    auto error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputKeys],
        isBlocking,
        offset,
        sizeof(DataType) * mNumberKeysRounded,
        mHostSpans.hResultFromGPUData()
    );
    if (error != CL_SUCCESS) return OperationStatus::DATA_DOWNLOAD_FAILED;
    error = CommandQueue.finish();
    using S = OperationStatus;
    return error == CL_SUCCESS ? S::OK : S::DATA_DOWNLOAD_FAILED;
}

    /// Downloads auxiliary buffers from device to host:
    /// histograms, globsum, input permutations, and output permutations.
    OperationStatus downloadIntermediate(cl::CommandQueue CommandQueue);

private:
    template <typename DataType>
    static std::string BuildPreamble()
    {
    using UnsignedType = typename std::make_unsigned<DataType>::type;

    const auto OFFSET { -std::numeric_limits<DataType>::min() };
    std::stringstream ss;
    ss << "#define DataType " << TypeNameString<DataType>::open_cl_name << std::endl
       << "#define UnsignedDataType " << TypeNameString<UnsignedType>::open_cl_name << std::endl
       << "#define OFFSET " << OFFSET << std::endl;
    return ss.str();
}
    /// Compiles build options for OpenCL kernel
    template <typename DataType>
    static std::string BuildOptions()
    {
    std::string options;
    //options += " -cl-opt-disable";
    options += " -cl-nv-verbose";
    // Compile options string
    {
        using Parameters = AlgorithmParameters<DataType>;
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
	void CopyDataToDevice(cl::CommandQueue CommandQueue){
    constexpr auto isBlocking = CL_FALSE;
    auto error = CL_SUCCESS;
    error = CommandQueue.enqueueWriteBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputKeys],
        isBlocking,
        0,
        sizeof(DataType) * mNumberKeysRounded,
        mHostSpans.hKeysData()
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueWriteBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputPermutations],
        isBlocking,
        0,
        sizeof(uint32_t) * mNumberKeysRounded,
        mHostSpans.hPermutData()
    );
    assert(error == CL_SUCCESS);
}
    template <typename DataType>
	void CopyDataFromDevice(cl::CommandQueue CommandQueue){
    constexpr auto isBlocking = CL_FALSE;
    constexpr auto offset = 0U;
    auto error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputKeys],
		isBlocking,
        offset,
		sizeof(DataType) * mNumberKeysRounded,
        mHostSpans.hResultFromGPUData()
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::InputPermutations],
		isBlocking,
        offset,
		sizeof(uint32_t) * mNumberKeysRounded,
        mHostSpans.hPermutData()
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::Histograms],
		isBlocking,
        offset,
		sizeof(uint32_t) * AlgorithmConfiguration::_RADIX * AlgorithmConfiguration::_NUM_GROUPS * AlgorithmConfiguration::_NUM_ITEMS_PER_GROUP,
        mHostSpans.hHistogramsData()
    );
    assert(error == CL_SUCCESS);

    error = CommandQueue.enqueueReadBuffer(
        mDeviceData->m_dMemoryMap[MemoryBuffer::Globsum],
		isBlocking,
        offset,
		sizeof(uint32_t)  * AlgorithmConfiguration::_NUM_HISTOSPLIT,
		mHostSpans.hGlobsumData()
    );
    assert(error == CL_SUCCESS);
}

    /// Device program, kernels and buffers
    std::shared_ptr<ComputeDeviceData> mDeviceData;
    /// Pointers to host memory buffers
    HostSpansProxy mHostSpans;

	// Runtime statistics GPU
    RuntimesGPU mRuntimesGPU{};

    // list of keys
    uint32_t mNumberKeysRounded{0U}; // next multiple of _ITEMS*_GROUPS

    /// log stream used for debugging
    std::ostream* mOutStream{nullptr};
};
