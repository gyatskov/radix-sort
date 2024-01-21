# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
)

set_source_files_properties("${Sources}"
    PROPERTIES
        COMPILE_FLAGS
            "$<IF:$<CXX_COMPILER_ID:MSVC>,\
            /W4;$<$<CONFIG:RELEASE>:/O2>,\
            -Wall;-Wextra;-Werror;-pedantic;>"
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
