cmake_minimum_required (VERSION 3.12.0)
project (radixsort)

# Add our modules to the path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/")

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include Common module
add_subdirectory (../Common ${CMAKE_BINARY_DIR}/Common)

# Define source files for this assignment
file(GLOB Sources   *.cpp)
file(GLOB Headers   *.h)
file(GLOB CLSources *.cl)

# TODO: Make this a library
add_executable (radixsort
	${Sources}
	${Headers}
	${CLSources}
)

# Link required libraries
target_link_libraries(radixsort
PRIVATE
    GPUCommon
)

target_compile_options(radixsort
PRIVATE
    -Werror -Wall -Wextra -pedantic
)


install(TARGETS radixsort RUNTIME DESTINATION bin)
install(FILES RadixSort.cl DESTINATION bin)
