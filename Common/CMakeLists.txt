# Enable C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FILE(GLOB CommonSources *.cpp)
FILE(GLOB CommonHeaders *.h)

CPMAddPackage(
  NAME OpenCL_SDK
  GIT_TAG aa33d56af666b7c20be71db639f27d6efab2f06a
  GITHUB_REPOSITORY "KhronosGroup/OpenCL-SDK"
  OPTIONS
    "BUILD_TESTING OFF"
    "BUILD_TESTS OFF"
    "BUILD_DOCS OFF"
    "OPENCL_SDK_BUILD_UTILITY_LIBRARIES ON" # Required by GPUCommon
    "OPENCL_SDK_BUILD_SAMPLES OFF"
    "OPENCL_SDK_TEST_SAMPLES OFF"
    "OPENCL_SDK_BUILD_OPENGL_SAMPLES OFF"
    "OPENCL_SDK_BUILD_VULKAN_SAMPLES OFF"
	"OPENCL_SDK_BUILD_CLINFO OFF"
)

set(OPENCL_VERSION 120)

add_library(GPUCommon STATIC
	${CommonSources}
	${CommonHeaders}
)

target_include_directories(GPUCommon
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(GPUCommon
# OpenCL::Headers and OpenCL::HeadersCpp are Khronos SDK-specific targets
# that provide <CL/opencl.hpp>.  Wrap them in BUILD_INTERFACE so they are
# only required when building from source (FetchContent / CPMAddPackage).
# When installed, the consumer's own OpenCL installation (found via
# FindOpenCL → OpenCL::OpenCL) provides the necessary include paths.
PUBLIC
    $<BUILD_INTERFACE:OpenCL::Headers>
    $<BUILD_INTERFACE:OpenCL::HeadersCpp>
    OpenCL::OpenCL
# OpenCL::Utils / OpenCL::UtilsCpp are only used inside GPUCommon's own .cpp
# files and do not appear in any public header.  BUILD_INTERFACE prevents
# them from leaking into the installed export set.
# They are used by tests.cpp
    $<BUILD_INTERFACE:OpenCL::Utils>
    $<BUILD_INTERFACE:OpenCL::UtilsCpp>
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

# ---------------------------------------------------------------------------
# Install rules — GPUCommon must be in the same export set as radixsortcl
# because it is a PUBLIC link dependency.
# ---------------------------------------------------------------------------
install(
    TARGETS GPUCommon
    EXPORT  radixsortclTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(
    DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION include/Common
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

