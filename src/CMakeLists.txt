# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include Common module
#add_subdirectory (../Common ${CMAKE_BINARY_DIR}/Common)

CPMAddPackage(
  NAME OpenCL_SDK
  VERSION 2023.12.14
  GITHUB_REPOSITORY "KhronosGroup/OpenCL-SDK"
  OPTIONS
    "BUILD_TESTING OFF"
    "BUILD_TESTS OFF"
    "BUILD_DOCS OFF"
    "OPENCL_SDK_BUILD_SAMPLES OFF"
    "OPENCL_SDK_TEST_SAMPLES OFF"
    "OPENCL_SDK_BUILD_OPENGL_SAMPLES OFF"
)

# Define source files
set(Sources
    RadixSortGPU.cpp
    CRadixSortTask.cpp
    ComputeDeviceData.cpp
    Dataset.cpp
    HostData.cpp
)
#file(GLOB Headers   *.h)
# TODO: Is this ever used?
file(GLOB CLSources *.cl)

add_library(radixsortcl
	${Sources}
	${CLSources}
)

target_include_directories(radixsortcl
PUBLIC
    ./
)

# Link required libraries
target_link_libraries(radixsortcl
PUBLIC
    GPUCommon
PRIVATE
    OpenCL::Headers
    OpenCL::HeadersCpp
    OpenCL::OpenCL
    OpenCL::Utils
    OpenCL::UtilsCpp
    #OpenCL::SDK
    #OpenCL::SDKCpp
)

string(
    APPEND compile_options
    "$<IF:$<CXX_COMPILER_ID:MSVC>,"
        "/W4;$<$<CONFIG:RELEASE>:/O2>,"
        "-Wall;-Wextra;-Werror;-pedantic;"
    ">"
)

target_compile_options(radixsortcl
PRIVATE
    ${compile_options}
)

install(
TARGETS
    radixsortcl
RUNTIME
DESTINATION
    bin
)
install(
FILES
    kernels/RadixSort.cl
DESTINATION
    bin
)
