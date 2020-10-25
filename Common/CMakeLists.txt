# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FILE(GLOB CommonSources *.cpp)
FILE(GLOB CommonHeaders *.h)

add_library(GPUCommon
	${CommonSources}
	${CommonHeaders}
)

target_compile_options(GPUCommon
PRIVATE
    -Werror -Wall -Wextra -pedantic
)
