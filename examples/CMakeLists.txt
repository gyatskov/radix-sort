add_executable(basic_sort basic_sort.cpp)

target_link_libraries(basic_sort
PRIVATE
    radixsortcl
    GPUCommon
)

# Copy the OpenCL kernel next to the example executable
add_custom_command(TARGET basic_sort POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/kernels/RadixSort.cl"
        "$<TARGET_FILE_DIR:basic_sort>/RadixSort.cl"
)
