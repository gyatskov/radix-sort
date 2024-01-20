# Testing library
include(CTest)
CPMAddPackage("gh:catchorg/Catch2@3.4.0")

# Tests need to be added as executables first
add_executable(tests
    tests.cpp
    CTestBase.cpp
)

target_compile_features(tests
PRIVATE
    cxx_std_17
)

# Should be linked to the main library, as well as the Catch2 testing library
target_link_libraries(tests
PRIVATE
    radixsortcl
    Catch2::Catch2WithMain
)

# Register tests for invocation via ctest
add_test(
    NAME tests
    COMMAND tests
    WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}/bin"
) # Command can be a target
