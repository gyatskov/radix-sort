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
    OpenCL::Utils
    OpenCL::UtilsCpp
)


string(
    APPEND compile_options
)
set_source_files_properties("${CommonSources}"
    PROPERTIES
        COMPILE_FLAGS
            "$<IF:$<CXX_COMPILER_ID:MSVC>,\
            /W4;$<$<CONFIG:RELEASE>:/O2>,\
            -Wall;-Wextra;-Werror;-pedantic;>"
)

