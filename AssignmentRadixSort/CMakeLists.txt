cmake_minimum_required (VERSION 3.10.0)
project (GPUComputing)

# Add our modules to the path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/")

# Enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "Enabling C++14 support")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include support for changing the working directory in Visual Studio
include(ChangeWorkingDirectory)

# Search for OpenCL and add paths
find_package( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )

# Include Common module
add_subdirectory (../Common ${CMAKE_BINARY_DIR}/Common)

# Define source files for this assignment
file(GLOB Sources   *.cpp)
file(GLOB Headers   *.h)
file(GLOB CLSources *.cl)
add_executable (Assignment
	${Sources}
	${Headers}
	${CLSources}
)

# Link required libraries
target_link_libraries(Assignment GPUCommon)
target_link_libraries(Assignment ${OPENCL_LIBRARIES})

if (WIN32)
	change_workingdir(Assignment ${CMAKE_SOURCE_DIR})
endif()
