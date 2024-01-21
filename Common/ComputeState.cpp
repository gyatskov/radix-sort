#include "ComputeState.h"
#include <iostream>
#include <algorithm>
#include <CL/Utils/Error.hpp>

cl::Platform ComputeState::platform() {
    return cl::Platform(device().getInfo<CL_DEVICE_PLATFORM>());
}

cl::Device ComputeState::device() {
    return m_CLDevices.front();
}

bool ComputeState::init() {
	//////////////////////////////////////////////////////
    {
        cl_int clError{-1};
        // 1. Enumerate OpenCL platforms
        clError = cl::Platform::get(&m_CLPlatforms);
        if(clError) {
            std::cerr << "Failed to enumerate devices: " << clError << "\n";
            return false;
        }

        // 2. find all available GPU devices
        cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

        for (auto platform : m_CLPlatforms)
        {
            decltype(m_CLDevices) devices;
            platform.getDevices(deviceType, &devices);

            std::copy(std::begin(devices), std::end(devices), std::back_inserter(m_CLDevices));
        }
    }

	if (m_CLDevices.size() == 0)
	{
		std::cerr << "No device of the selected type with OpenCL support was found.\n";
		return false;
	}

	// Printing platform and device data.
    {
        auto plat = platform();
        std::cout << "OpenCL platform:\n\n"
            << "Name    " << plat.getInfo<CL_PLATFORM_NAME>() << "\n"
            << "Vendor  " << plat.getInfo<CL_PLATFORM_VENDOR>() << "\n"
            << "Version " << plat.getInfo<CL_PLATFORM_VERSION>()<< "\n"
            << "Profile " << plat.getInfo<CL_PLATFORM_PROFILE>() << "\n"
            << "\n";

        auto dev = device();
        std::cout << "Device:\n\n"
            << "Name         " << dev.getInfo<CL_DEVICE_NAME>() << "\n"
            << "Vendor       " << dev.getInfo<CL_DEVICE_VENDOR>() << "\n"
            << "Version      " << dev.getInfo<CL_DRIVER_VERSION>() << "\n"
            << "Local Memory " << dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << "\n";
        std::cout << "\n*********************************\n\n";
    }

    {
        cl_int clError{-1};
        const auto properties = nullptr;
        const auto callback = nullptr;
        const auto userData = nullptr;

        m_CLContext = cl::Context(
            device(),
            properties,
            callback,
            userData,
            &clError
        );
        if(clError) {
            std::cerr<<cl::util::Error(clError, "Failed to create OpenCL context.").what()<<"\n";
            return false;
        }
    }

	// Finally, create a command queue. All the asynchronous commands to the device will be issued
	// from the CPU into this queue. This way the host program can continue the execution until some results
	// from that device are needed.

    {
        cl_int clError{-1};
        const auto properties = 0;
        m_CLCommandQueue = cl::CommandQueue(
                m_CLContext,
                device(),
                properties,
                &clError
        );
        if(clError) {
            std::cerr<<cl::util::Error(clError, "Failed to create the command queue in the context").what()<<"\n";
            return false;
        }
    }

	return true;
}

