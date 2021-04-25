# Testing library
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.9.1)
FetchContent_MakeAvailable(catch2)
# Adds Catch2::Catch2

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
    Catch2::Catch2
)

# If you register a test, then ctest and make test will run it. You can also run
# examples and check the output, as well.
add_test(
    NAME tests
    COMMAND tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
) # Command can be a target
