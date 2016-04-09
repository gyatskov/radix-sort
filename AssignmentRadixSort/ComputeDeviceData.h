#pragma once

#include "../Common/CLUtil.h"
#include <vector>
#include <map>
#include <string>

struct ComputeDeviceData {
    ComputeDeviceData();
    ~ComputeDeviceData();

    //OpenCL program and kernels
    cl_program			     m_Program;
    std::vector<std::string> kernelNames;
    std::vector<std::string> alternatives;

    std::map<std::string, cl_kernel> m_kernelMap;
    std::map<std::string, cl_mem>    m_dMemoryMap; // NOTE: not used yet

    cl_mem m_dHistograms;                // histograms on the GPU

    cl_mem m_dGlobsum;
    cl_mem m_dTemp;  // in case where the sum is not needed

    cl_mem m_dInKeys;
    cl_mem m_dOutKeys;

    cl_mem m_dInPermut;
    cl_mem m_dOutPermut;
};
