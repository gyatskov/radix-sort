#include "ComputeDeviceData.h"

#include "../Common/CLUtil.h"

#include <cstdint>

template <typename DataType>
ComputeDeviceData<DataType>::ComputeDeviceData(
    cl_context Context,
    size_t buffer_size
    )
    : m_Program(nullptr)
{
    kernelNames.emplace_back("histogram");
    kernelNames.emplace_back("scanhistograms");
    kernelNames.emplace_back("pastehistograms");
    kernelNames.emplace_back("reorder");

    alternatives.emplace_back("RadixSort_01");

	// allocate device resources
    const auto createBufferAndCheck = [Context](auto& target, auto sizeInBytes) {
        cl_int clError;

        TODO("Consider using CL_MEM_USE_HOST_PTR for user-provided memory");
        target = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeInBytes, nullptr, &clError);
        constexpr auto ERROR_STRING = "Error allocating device array";
        V_RETURN_CL(clError, ERROR_STRING);
    };

    createBufferAndCheck(m_dInKeys, sizeof(DataType) * buffer_size);

	createBufferAndCheck(m_dOutKeys, sizeof(DataType) * buffer_size);

	createBufferAndCheck(m_dInPermut, sizeof(uint32_t) * buffer_size);
	createBufferAndCheck(m_dOutPermut, sizeof(uint32_t) * buffer_size);

	// allocate the histogram on the GPU
	createBufferAndCheck(m_dHistograms, sizeof(uint32_t) * Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP);

	// allocate the auxiliary histogram on GPU
	createBufferAndCheck(m_dGlobsum, sizeof(uint32_t) * Parameters::_NUM_HISTOSPLIT);

	// temporary vector when the sum is not needed
	createBufferAndCheck(m_dTemp, sizeof(uint32_t) * Parameters::_NUM_HISTOSPLIT);
}

template <typename DataType>
ComputeDeviceData<DataType>::~ComputeDeviceData()
{
    SAFE_RELEASE_MEMOBJECT(m_dInKeys);
    SAFE_RELEASE_MEMOBJECT(m_dOutKeys);
    SAFE_RELEASE_MEMOBJECT(m_dInPermut);
    SAFE_RELEASE_MEMOBJECT(m_dOutPermut);

    for (auto& kernel : m_kernelMap) {
        SAFE_RELEASE_KERNEL(kernel.second);
    }

    SAFE_RELEASE_PROGRAM(m_Program);
}

// Specialize ComputeDeviceData for exactly these four types.
template struct ComputeDeviceData < int32_t >;
template struct ComputeDeviceData < int64_t >;
template struct ComputeDeviceData < uint32_t >;
template struct ComputeDeviceData < uint64_t >;
