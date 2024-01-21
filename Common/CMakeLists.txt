# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FILE(GLOB CommonSources *.cpp)
FILE(GLOB CommonHeaders *.h)

CPMAddPackage(
  NAME OpenCL_SDK
  GIT_TAG fix-whereami-dep
  GITHUB_REPOSITORY "gyatskov/OpenCL-SDK"
  OPTIONS
    "BUILD_TESTING OFF"
    "BUILD_TESTS OFF"
    "BUILD_DOCS OFF"
    "OPENCL_SDK_BUILD_SAMPLES OFF"
    "OPENCL_SDK_TEST_SAMPLES OFF"
    "OPENCL_SDK_BUILD_OPENGL_SAMPLES OFF"
)

set(OPENCL_VERSION 120)

add_library(GPUCommon
	${CommonSources}
	${CommonHeaders}
)

target_link_libraries(GPUCommon
PUBLIC
    OpenCL::Headers
    OpenCL::HeadersCpp
    OpenCL::OpenCL
    OpenCL::Utils
    OpenCL::UtilsCpp
)

target_compile_definitions(GPUCommon
PUBLIC
    CL_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
    CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
    CL_HPP_MINIMUM_OPENCL_VERSION=${OPENCL_VERSION}
    CL_HPP_ENABLE_EXCEPTIONS
)

set_source_files_properties("${CommonSources}"
    PROPERTIES
        COMPILE_FLAGS
            "$<IF:$<CXX_COMPILER_ID:MSVC>,\
            /W4;$<$<CONFIG:RELEASE>:/O2>,\
            -Wall;-Wextra;-Werror;-pedantic;>"
)

