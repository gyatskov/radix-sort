# ── basic_sort ───────────────────────────────────────────────────────────
add_executable(basic_sort basic_sort.cpp)

target_link_libraries(basic_sort
PRIVATE
    radixsortcl
    GPUCommon
)

add_custom_command(TARGET basic_sort POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/kernels/RadixSort.cl"
        "$<TARGET_FILE_DIR:basic_sort>/RadixSort.cl"
)

add_custom_command(TARGET basic_sort POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:basic_sort>
        $<TARGET_FILE_DIR:basic_sort>
    COMMAND_EXPAND_LISTS
)