cmake_minimum_required (VERSION 2.8.3) 
project (GPUComputing) 

# Add our modules to the path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/")

if (WIN32)
else ()
	# Enable C++11
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	MESSAGE(STATUS "Enabling C++11 support")
endif()

# Include support for changing the working directory in Visual Studio
include(ChangeWorkingDirectory)

# Search for OpenCL and add paths
find_package( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )

# Include Common module
add_subdirectory (../Common ${CMAKE_BINARY_DIR}/Common) 

# Define source files for this assignment
FILE(GLOB Sources *.cpp)
FILE(GLOB Headers *.h)
FILE(GLOB CLSources *.cl)
ADD_EXECUTABLE (Assignment 
	${Sources}
	${Headers}
	${CLSources}
	)

# Link required libraries
target_link_libraries(Assignment ${OPENCL_LIBRARIES})
target_link_libraries(Assignment GPUCommon)

if (WIN32)
	change_workingdir(Assignment ${CMAKE_SOURCE_DIR})
endif()
