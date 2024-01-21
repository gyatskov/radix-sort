#include "ComputeDeviceData.h"

#include <CL/Utils/Error.hpp>

#include <cstdint>
#include <iostream>
#include <string>

template <typename DataType>
ComputeDeviceData<DataType>::ComputeDeviceData(
    cl::Context Context,
    size_t buffer_size
)
{
    kernelNames.emplace_back("histogram");
    kernelNames.emplace_back("scanhistograms");
    kernelNames.emplace_back("pastehistograms");
    kernelNames.emplace_back("reorder");

	// allocate device resources
    const auto createBufferAndCheck = [Context](
            auto& target,
            auto sizeInBytes) {
        cl_int clError{CL_SUCCESS};

#pragma message("Consider using CL_MEM_USE_HOST_PTR for user-provided memory")
        constexpr auto hostPtr = nullptr;
        target = cl::Buffer(
            Context,
            CL_MEM_READ_WRITE,
            sizeInBytes,
            hostPtr,
            &clError
        );
        if(clError) {
            constexpr auto ERROR_STRING = "Error allocating device array";
            std::cerr<<cl::util::Error(clError, ERROR_STRING).what()<<"\n";
        }
        return clError;
    };

    createBufferAndCheck(
        m_dMemoryMap["inputKeys"],
        sizeof(DataType) * buffer_size
    );

	createBufferAndCheck(
        m_dMemoryMap["outputKeys"],
        sizeof(DataType) * buffer_size
    );

	createBufferAndCheck(
        m_dMemoryMap["inputPermutations"],
        sizeof(uint32_t) * buffer_size
    );
	createBufferAndCheck(
        m_dMemoryMap["outputPermutations"],
        sizeof(uint32_t) * buffer_size
    );

	// allocate the histogram on the GPU
	createBufferAndCheck(
        m_dMemoryMap["histograms"],
        sizeof(uint32_t) * Parameters::_RADIX * Parameters::_NUM_ITEMS
    );

	// allocate the auxiliary histogram on GPU
	createBufferAndCheck(
        m_dMemoryMap["globsum"],
        sizeof(uint32_t) * Parameters::_NUM_HISTOSPLIT
    );

	// temporary vector when the sum is not needed
	createBufferAndCheck(
        m_dMemoryMap["temp"],
        sizeof(uint32_t) * Parameters::_NUM_HISTOSPLIT
    );
}

// Specialize ComputeDeviceData for exactly these four types.
template struct ComputeDeviceData < int32_t >;
template struct ComputeDeviceData < int64_t >;
template struct ComputeDeviceData < uint32_t >;
template struct ComputeDeviceData < uint64_t >;
