#include "ComputeState.h"
#include <algorithm>

cl::Platform ComputeState::platform() {
    return cl::Platform(device().getInfo<CL_DEVICE_PLATFORM>());
}
cl::Device ComputeState::device() {
    return m_CLDevices.front();
}
bool ComputeState::init() {
	//////////////////////////////////////////////////////
	// 1. Enumerate OpenCL platforms
    cl::Platform::get(&m_CLPlatforms);


	// 2. find all available GPU devices
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

	for (auto platform : m_CLPlatforms)
	{
        decltype(m_CLDevices) devices;
        platform.getDevices(deviceType, &devices);

        std::copy(std::begin(devices), std::end(devices), std::end(m_CLDevices));
	}

	if (m_CLDevices.size() == 0)
	{
		std::cerr << "No device of the selected type with OpenCL support was found.\n";
		return false;
	}

	// Printing platform and device data.
    {
        auto p = platform();
        std::cout << "OpenCL platform:\n\n";

        std::cout<< "Name" <<  p.getInfo<CL_PLATFORM_NAME>() << "\n"
                 << "Vendor" <<  p.getInfo<CL_PLATFORM_VENDOR>() << "\n"
                 << "Version" <<  p.getInfo<CL_PLATFORM_VERSION>()<< "\n"
                 << "Profile" <<  p.getInfo<CL_PLATFORM_PROFILE>() << "\n";

        auto d = device();
        std::cout << std::endl << "Device:\n\n"
            << "Name" <<d.getInfo< CL_DEVICE_NAME>() << "\n"
            << "Vendor" <<d.getInfo< CL_DEVICE_VENDOR>() << "\n"
            << "Version" <<d.getInfo< CL_DRIVER_VERSION>() << "\n"
            << "Local Memory" <<d.getInfo< CL_DEVICE_LOCAL_MEM_SIZE>() << "\n";
        std::cout << std::endl << "******************************\n\n";
    }

    {
        cl_int clError{};
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
        V_RETURN_FALSE_CL(clError, "Failed to create OpenCL context.");
    }

	// Finally, create a command queue. All the asynchronous commands to the device will be issued
	// from the CPU into this queue. This way the host program can continue the execution until some results
	// from that device are needed.

    {
        cl_int clError{};
        const auto properties = 0;
        m_CLCommandQueue = cl::CommandQueue(m_CLContext, device(), properties, &clError);
        V_RETURN_FALSE_CL(clError, "Failed to create the command queue in the context");
    }

	return true;
}
#undef PRINT_INFO

