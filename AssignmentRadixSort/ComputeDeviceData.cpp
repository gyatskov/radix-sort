#include "ComputeDeviceData.h"
#include "../Common/CLUtil.h"

ComputeDeviceData::ComputeDeviceData() :
    m_Program(NULL)
{
    kernelNames.emplace_back("histogram");
    kernelNames.emplace_back("scanhistograms");
    kernelNames.emplace_back("pastehistograms");
    kernelNames.emplace_back("reorder");
    kernelNames.emplace_back("transpose");

    alternatives.emplace_back("RadixSort_01");
}

ComputeDeviceData::~ComputeDeviceData() {
    SAFE_RELEASE_MEMOBJECT(m_dInKeys);
    SAFE_RELEASE_MEMOBJECT(m_dOutKeys);
    SAFE_RELEASE_MEMOBJECT(m_dInPermut);
    SAFE_RELEASE_MEMOBJECT(m_dOutPermut);

    for (auto& kernel : m_kernelMap) {
        SAFE_RELEASE_KERNEL(kernel.second);
    }

    SAFE_RELEASE_PROGRAM(m_Program);
}