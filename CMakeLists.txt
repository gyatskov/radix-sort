cmake_minimum_required (VERSION 3.12.0)
project (radixsort
    VERSION 0.1
    DESCRIPTION "OpenCL 1.2 optimized radix sort implementation"
    LANGUAGES CXX
)

# Disable std=g++xx, use standard
set(CMAKE_CXX_EXTENSIONS OFF)

# Add custom modules to the path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)
include(FetchContent)

add_subdirectory(src)
add_subdirectory(Common ${CMAKE_BINARY_DIR}/Common)
add_subdirectory(tests)

