# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include Common module
#add_subdirectory (../Common ${CMAKE_BINARY_DIR}/Common)

# Define source files
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

string(
    APPEND compile_options
    "$<IF:$<CXX_COMPILER_ID:MSVC>,"
        "/W4;$<$<CONFIG:RELEASE>:/O2>,"
        "-Wall;-Wextra;-Werror;-pedantic;"
    ">"
)

target_compile_options(radixsort
PRIVATE
    ${compile_options}
)

install(TARGETS radixsort RUNTIME DESTINATION bin)
install(FILES RadixSort.cl DESTINATION bin)
