# Define source files
set(Sources
    RadixSortGPU.cpp
    ComputeDeviceData.cpp
    Dataset.cpp
    HostData.cpp
)

add_library(radixsortcl STATIC
    ${Sources}
)

target_include_directories(radixsortcl
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
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

# ---------------------------------------------------------------------------
# Pre-compile RadixSort.cl for each supported data type.
#
# The compile_opencl_kernels() function (from cmake/CompileOpenCLKernel.cmake)
# will:
#   1. Detect available offline OpenCL tools (clang -target spir64 / clspv /
#      clang+llvm-spirv for SPIR-V; clang for native binary).
#   2. Emit add_custom_command rules that produce per-type kernel files
#      in ${CMAKE_CURRENT_BINARY_DIR}/kernels/.
#   3. Create a radixsortcl_kernels aggregate target that radixsortcl depends on.
#   4. Set RADIXSORTCL_KERNEL_DIR / RADIXSORTCL_KERNEL_EXT compile definitions
#      so RadixSortGPU.cpp knows where to look at runtime.
#
# If no offline compiler is found the function emits a warning and returns
# without adding any rules; the library will fall back to runtime source
# compilation as before.
# ---------------------------------------------------------------------------
include(CompileOpenCLKernel)

compile_opencl_kernels(
    TARGET        radixsortcl
    KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/RadixSort.cl"
    OUTPUT_DIR    "${CMAKE_CURRENT_BINARY_DIR}/kernels"
    DATA_TYPES    int32 uint32 int64 uint64
)

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------

# Static library and headers
install(
    TARGETS radixsortcl
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# OpenCL source kernel (always installed as a final fallback for runtime JIT)
install(
    FILES kernels/RadixSort.cl
    DESTINATION share/radixsortcl/kernels
)

# Pre-compiled kernel binaries (installed when offline compilation succeeded)
if(DEFINED RADIXSORTCL_COMPILED_KERNELS AND RADIXSORTCL_COMPILED_KERNELS)
    install(
        FILES ${RADIXSORTCL_COMPILED_KERNELS}
        DESTINATION share/radixsortcl/kernels
    )
endif()
