# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include Common module
#add_subdirectory (../Common ${CMAKE_BINARY_DIR}/Common)

# Define source files
set(Sources
    RadixSortGPU.cpp
    CRadixSortTask.cpp
    ComputeDeviceData.cpp
    Dataset.cpp
    HostData.cpp
)
#file(GLOB Headers   *.h)
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
PRIVATE
    GPUCommon
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

# TODO: Check if this makes sense at all
install(TARGETS radixsortcl RUNTIME DESTINATION bin)
install(FILES RadixSort.cl DESTINATION bin)
