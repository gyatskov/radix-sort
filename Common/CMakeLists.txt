# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Search for OpenCL and add paths
find_package(OpenCL 1.2 REQUIRED)

FILE(GLOB CommonSources *.cpp)
FILE(GLOB CommonHeaders *.h)

add_library(GPUCommon
	${CommonSources}
	${CommonHeaders}
)

target_link_libraries(GPUCommon
PUBLIC
    OpenCL::OpenCL
)


string(
    APPEND compile_options
    "$<IF:$<CXX_COMPILER_ID:MSVC>,"
        "/W4;$<$<CONFIG:RELEASE>:/O2>,"
        "-Wall;-Wextra;-Werror;-pedantic;"
    ">"
)

target_compile_options(GPUCommon
PRIVATE
    ${compile_options}
)
