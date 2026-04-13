cmake_minimum_required (VERSION 4.0)
project (radixsort
    VERSION 0.2
    DESCRIPTION "OpenCL 1.2 optimized radix sort implementation"
    LANGUAGES CXX
)

include(CTest)
option(RADIXSORT_BUILD_EXAMPLES "Build examples" ON)

# Add custom modules to the path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# When consumed via FetchContent / CPMAddPackage the parent project needs
# CompileOpenCLKernel.cmake too.  Propagate the cmake/ directory up so that
# after FetchContent_MakeAvailable(radixsort) the parent can
#   include(CompileOpenCLKernel)
# without knowing where the sources were cloned.
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
endif()

# Enable C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Disable std=g++xx, use standard
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(cmake/CPM.cmake)

add_subdirectory(src)
add_subdirectory(Common ${CMAKE_BINARY_DIR}/Common)
if (RADIXSORT_BUILD_EXAMPLES)
    message(STATUS "Building examples")
    add_subdirectory(examples)
endif()
if (BUILD_TESTING)
    message(STATUS "Building tests")
    add_subdirectory(tests)
endif()

