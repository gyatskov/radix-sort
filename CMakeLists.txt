cmake_minimum_required (VERSION 4.0)
project (radixsort
    VERSION 0.2
    DESCRIPTION "OpenCL 1.2 optimized radix sort implementation"
    LANGUAGES CXX
)

include(CTest)

# Add custom modules to the path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Enable C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Disable std=g++xx, use standard
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(cmake/CPM.cmake)

add_subdirectory(src)
add_subdirectory(Common ${CMAKE_BINARY_DIR}/Common)
add_subdirectory(examples)
if (BUILD_TESTING)
    message(STATUS "Building tests")
    add_subdirectory(tests)
endif()

