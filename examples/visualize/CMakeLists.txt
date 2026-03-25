
# ── visualize (optional – requires Vulkan SDK with glslc) ───────────────
find_package(Vulkan QUIET)
if(Vulkan_FOUND AND TARGET Vulkan::glslc)
    CPMAddPackage(
        NAME glfw
        GITHUB_REPOSITORY glfw/glfw
        GIT_TAG 3.4
        OPTIONS
            "GLFW_BUILD_DOCS OFF"
            "GLFW_BUILD_TESTS OFF"
            "GLFW_BUILD_EXAMPLES OFF"
            "GLFW_INSTALL OFF"
    )
    CPMAddPackage(
        NAME vk-bootstrap
        GITHUB_REPOSITORY charles-lunarg/vk-bootstrap
        GIT_TAG v1.4.321
    )

    add_executable(visualize visualize.cpp)

    target_link_libraries(visualize
    PRIVATE
        radixsortcl
        GPUCommon
        Vulkan::Vulkan
        glfw
        vk-bootstrap::vk-bootstrap
    )

    # Copy OpenCL kernel next to the executable
    add_custom_command(TARGET visualize POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/src/kernels/RadixSort.cl"
            "$<TARGET_FILE_DIR:visualize>/RadixSort.cl"
    )

    # Copy runtime DLLs next to the executable
    add_custom_command(TARGET visualize POST_BUILD
        #        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        #            $<TARGET_RUNTIME_DLLS:visualize>
        #            $<TARGET_FILE_DIR:visualize>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:glfw>
            $<TARGET_FILE_DIR:visualize>
        COMMAND_EXPAND_LISTS
    )

    # Compile GLSL shaders to SPIR-V next to the executable
    add_custom_command(TARGET visualize POST_BUILD
        COMMAND Vulkan::glslc
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/visualize.vert"
            -o "$<TARGET_FILE_DIR:visualize>/visualize.vert.spv"
        COMMAND Vulkan::glslc
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/visualize.frag"
            -o "$<TARGET_FILE_DIR:visualize>/visualize.frag.spv"
        COMMAND Vulkan::glslc
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/overlay.vert"
            -o "$<TARGET_FILE_DIR:visualize>/overlay.vert.spv"
        COMMAND Vulkan::glslc
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/overlay.frag"
            -o "$<TARGET_FILE_DIR:visualize>/overlay.frag.spv"
        COMMENT "Compiling visualize shaders to SPIR-V"
    )
else()
    message(STATUS "Vulkan SDK with glslc not found — skipping visualize example")
endif()
