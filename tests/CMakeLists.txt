# Testing library
include(CTest)
CPMAddPackage(
    NAME Catch2
    GIT_TAG v3.13.0
    GITHUB_REPOSITORY catchorg/Catch2
    OPTIONS "BUILD_SHARED_LIBS OFF"
)

# Tests need to be added as executables first
add_executable(tests
    tests.cpp
    CTestBase.cpp
)

# Should be linked to the main library, as well as the Catch2 testing library
target_link_libraries(tests
PRIVATE
    radixsortcl
    GPUCommon
    Catch2::Catch2WithMain
)

enable_testing()
# Copy the OpenCL kernel next to the test executable so it can be found at runtime
add_custom_command(TARGET tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/kernels/RadixSort.cl"
        "$<TARGET_FILE_DIR:tests>/RadixSort.cl"
)

# Register tests for invocation via ctest
add_test(
    NAME tests
    COMMAND tests
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:tests>"
) # Command can be a target
