# CompileOpenCLKernel.cmake
#
# Provides compile_opencl_kernels() to pre-compile RadixSort.cl for each
# supported data type at build time.
#
# Output format selection:
#
#   SPIR-V  (.spv) — preferred when available.
#     Tool search order:
#       1. clang -target spir64  +  llvm-spirv   (clang >= 14 + LLVM translator)
#       2. clspv                                  (Khronos offline compiler)
#       -- If neither produces SPIR-V, fall through to SPIR --
#
#   SPIR    (.bc)  — LLVM bitcode targeting spir/spir64 (cl_khr_spir).
#     Loaded at runtime via clCreateProgramWithBinary + cl_khr_spir.
#       1. clang -target spir64 -emit-llvm -c
#       -- If clang not found, no pre-compiled binary is produced --
#
# Public API:
#
#   compile_opencl_kernels(
#       TARGET          <cmake-target>
#       KERNEL_SOURCE   <path/to/file.cl>   # absolute path
#       OUTPUT_DIR      <dir>               # defaults to ${CMAKE_CURRENT_BINARY_DIR}/kernels
#       DATA_TYPES      int32 uint32 int64 uint64
#       COMPILE_DEFS    <def=val> ...       # extra -D flags (without -D prefix)
#   )
#
# For each type the function:
#   • emits add_custom_command rules that produce
#       <OUTPUT_DIR>/RadixSort_<type>.spv   (SPIR-V)  or
#       <OUTPUT_DIR>/RadixSort_<type>.bc    (SPIR)
#   • creates a <TARGET>_kernels aggregate target that <TARGET> depends on
#   • sets RADIXSORTCL_KERNEL_DIR and RADIXSORTCL_KERNEL_EXT definitions on
#     <TARGET> so the C++ code knows where to look at runtime
#
# Parent-scope variables set after the call:
#   RADIXSORTCL_COMPILED_KERNELS  — list of output files (for install rules)
#   RADIXSORTCL_KERNEL_EXT        — "spv" or "bc"
#   RADIXSORTCL_KERNEL_MODE       — "SPIRV" or "SPIR"

cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Internal: locate tools once, cache the results
# ---------------------------------------------------------------------------
function(_rsort_find_tools)
    if(DEFINED CACHE{_RSORT_TOOLS_FOUND})
        return()
    endif()

    # --- clang ---
    find_program(CLANG_EXECUTABLE
        NAMES clang clang-20 clang-19 clang-18 clang-17 clang-16 clang-15 clang-14
        DOC "clang used for offline OpenCL → SPIR / SPIR-V compilation")

    # --- llvm-spirv ---
    find_program(LLVM_SPIRV_EXECUTABLE
        NAMES llvm-spirv llvm-spirv-20 llvm-spirv-19 llvm-spirv-18
              llvm-spirv-17 llvm-spirv-16 llvm-spirv-15 llvm-spirv-14
        DOC "LLVM SPIR-V translator (llvm-spirv)")

    # --- clspv ---
    find_program(CLSPV_EXECUTABLE
        NAMES clspv
        DOC "Khronos clspv OpenCL → SPIR-V compiler")

    # ----------------------------------------------------------------
    # Determine best available mode
    # ----------------------------------------------------------------
    set(_mode "NONE")

    # Helper: test whether clang accepts -target spir64 for OpenCL
    macro(_rsort_clang_supports_spir64 _result_var)
        # Write a minimal .cl file for the test — /dev/null doesn't work on
        # all platforms, and CMake's execute_process doesn't use a shell.
        set(_test_cl "${CMAKE_CURRENT_BINARY_DIR}/_rsort_spir64_test.cl")
        if(NOT EXISTS "${_test_cl}")
            file(WRITE "${_test_cl}" "__kernel void _t(__global int*p){*p=0;}\n")
        endif()
        execute_process(
            COMMAND "${CLANG_EXECUTABLE}"
                    -target spir64-unknown-unknown
                    -x cl -cl-std=CL1.2
                    -fsyntax-only
                    "${_test_cl}"
            RESULT_VARIABLE "${_result_var}"
            OUTPUT_QUIET ERROR_QUIET
        )
    endmacro()

    # SPIR-V via clang + llvm-spirv (most common on Linux)
    if(CLANG_EXECUTABLE AND LLVM_SPIRV_EXECUTABLE)
        _rsort_clang_supports_spir64(_clang_spir64_rc)
        if(_clang_spir64_rc EQUAL 0)
            set(_mode "SPIRV_CLANG_LLVMSPIRV")
        endif()
    endif()

    # SPIR-V via clspv
    if(_mode STREQUAL "NONE" AND CLSPV_EXECUTABLE)
        set(_mode "SPIRV_CLSPV")
    endif()

    # SPIR (legacy LLVM bitcode) via clang -target spir64
    if(_mode STREQUAL "NONE" AND CLANG_EXECUTABLE)
        _rsort_clang_supports_spir64(_clang_spir64_rc2)
        if(_clang_spir64_rc2 EQUAL 0)
            set(_mode "SPIR_CLANG")
        endif()
    endif()

    set(_RSORT_KERNEL_MODE "${_mode}" CACHE INTERNAL "OpenCL kernel compile mode")
    set(_RSORT_TOOLS_FOUND TRUE       CACHE INTERNAL "Tool detection done")

    if(_mode STREQUAL "NONE")
        message(STATUS
            "[radixsortcl] No offline OpenCL compiler found "
            "(clang+llvm-spirv, clspv, or clang with -target spir64). "
            "Kernels will be compiled from source at runtime.")
    else()
        message(STATUS "[radixsortcl] OpenCL offline compile mode: ${_mode}")
        if(CLANG_EXECUTABLE)
            message(STATUS "[radixsortcl]   clang      : ${CLANG_EXECUTABLE}")
        endif()
        if(LLVM_SPIRV_EXECUTABLE)
            message(STATUS "[radixsortcl]   llvm-spirv : ${LLVM_SPIRV_EXECUTABLE}")
        endif()
        if(CLSPV_EXECUTABLE)
            message(STATUS "[radixsortcl]   clspv      : ${CLSPV_EXECUTABLE}")
        endif()
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Internal: per-type kernel defines (must match Parameters.h / CLTypeInformation.h)
# ---------------------------------------------------------------------------
function(_rsort_type_defines TYPE_NAME OUT_DEFS)
    if(TYPE_NAME STREQUAL "int32")
        set(${OUT_DEFS}
            "-DDataType=int"
            "-DUnsignedDataType=uint"
            "-DOFFSET=2147483648"    # -INT32_MIN = 2^31
            PARENT_SCOPE)
    elseif(TYPE_NAME STREQUAL "uint32")
        set(${OUT_DEFS}
            "-DDataType=uint"
            "-DUnsignedDataType=uint"
            "-DOFFSET=0"
            PARENT_SCOPE)
    elseif(TYPE_NAME STREQUAL "int64")
        set(${OUT_DEFS}
            "-DDataType=long"
            "-DUnsignedDataType=ulong"
            "-DOFFSET=9223372036854775808UL"  # -INT64_MIN = 2^63
            PARENT_SCOPE)
    elseif(TYPE_NAME STREQUAL "uint64")
        set(${OUT_DEFS}
            "-DDataType=ulong"
            "-DUnsignedDataType=ulong"
            "-DOFFSET=0"
            PARENT_SCOPE)
    else()
        message(FATAL_ERROR
            "[radixsortcl] Unknown kernel data type '${TYPE_NAME}'. "
            "Supported: int32 uint32 int64 uint64")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Internal: algorithm parameter defines (mirror AlgorithmConfiguration /
#           AlgorithmParameters in Parameters.h)
# ---------------------------------------------------------------------------
function(_rsort_algorithm_defines TYPE_NAME OUT_DEFS)
    # _NUM_BITS_PER_RADIX = 4  =>  _RADIX = 16
    # _NUM_ITEMS_PER_GROUP = 64, _NUM_GROUPS = 16  =>  _NUM_ITEMS = 1024
    # _NUM_HISTOSPLIT = 512
    # _TOTALBITS: 32 for 32-bit types, 64 for 64-bit types
    if(TYPE_NAME MATCHES "64")
        set(_totalbits 64)
        set(_passes    16)   # 64 / 4
    else()
        set(_totalbits 32)
        set(_passes    8)    # 32 / 4
    endif()
    set(${OUT_DEFS}
        "-D_ITEMS=64"
        "-D_GROUPS=16"
        "-D_HISTOSPLIT=512"
        "-D_TOTALBITS=${_totalbits}"
        "-D_BITS=4"
        "-D_N=33554432"       # 2^25 = _NUM_MAX_INPUT_ELEMS
        "-D_RADIX=16"
        "-D_PASS=${_passes}"
        "-D_HISTOSIZE=16384"  # 64 * 16 * 16
        PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Internal: emit add_custom_command for one type in SPIRV_CLANG_LLVMSPIRV mode
# ---------------------------------------------------------------------------
function(_rsort_cmd_spirv_clang_llvmspirv KERNEL_SOURCE TYPE_NAME OUTPUT_SPV EXTRA_DEFS)
    _rsort_type_defines("${TYPE_NAME}" _type_defs)
    _rsort_algorithm_defines("${TYPE_NAME}" _algo_defs)

    get_filename_component(_out_dir "${OUTPUT_SPV}" DIRECTORY)
    set(_bc_file "${_out_dir}/RadixSort_${TYPE_NAME}.bc")

    add_custom_command(
        OUTPUT "${OUTPUT_SPV}"
        # Step 1: clang → LLVM bitcode targeting spir64
        COMMAND "${CLANG_EXECUTABLE}"
            -target spir64-unknown-unknown
            -x cl -cl-std=CL1.2 -O2
            -emit-llvm -c
            ${_type_defs} ${_algo_defs} ${EXTRA_DEFS}
            "${KERNEL_SOURCE}"
            -o "${_bc_file}"
        # Step 2: llvm-spirv → SPIR-V
        COMMAND "${LLVM_SPIRV_EXECUTABLE}"
            "${_bc_file}"
            -o "${OUTPUT_SPV}"
        DEPENDS "${KERNEL_SOURCE}"
        BYPRODUCTS "${_bc_file}"
        COMMENT
            "[radixsortcl] Compiling ${TYPE_NAME} kernel → SPIR-V (clang + llvm-spirv)"
        VERBATIM
    )
endfunction()

# ---------------------------------------------------------------------------
# Internal: emit add_custom_command for one type in SPIRV_CLSPV mode
# ---------------------------------------------------------------------------
function(_rsort_cmd_spirv_clspv KERNEL_SOURCE TYPE_NAME OUTPUT_SPV EXTRA_DEFS)
    _rsort_type_defines("${TYPE_NAME}" _type_defs)
    _rsort_algorithm_defines("${TYPE_NAME}" _algo_defs)

    add_custom_command(
        OUTPUT "${OUTPUT_SPV}"
        COMMAND "${CLSPV_EXECUTABLE}"
            --spv-version=1.0
            -cl-std=CL1.2
            ${_type_defs} ${_algo_defs} ${EXTRA_DEFS}
            -o "${OUTPUT_SPV}"
            "${KERNEL_SOURCE}"
        DEPENDS "${KERNEL_SOURCE}"
        COMMENT
            "[radixsortcl] Compiling ${TYPE_NAME} kernel → SPIR-V (clspv)"
        VERBATIM
    )
endfunction()

# ---------------------------------------------------------------------------
# Internal: emit add_custom_command for one type in SPIR_CLANG mode
#           Produces LLVM bitcode for spir64 target, loaded via cl_khr_spir
# ---------------------------------------------------------------------------
function(_rsort_cmd_spir_clang KERNEL_SOURCE TYPE_NAME OUTPUT_BC EXTRA_DEFS)
    _rsort_type_defines("${TYPE_NAME}" _type_defs)
    _rsort_algorithm_defines("${TYPE_NAME}" _algo_defs)

    add_custom_command(
        OUTPUT "${OUTPUT_BC}"
        COMMAND "${CLANG_EXECUTABLE}"
            -target spir64-unknown-unknown
            -x cl -cl-std=CL1.2 -O2
            -emit-llvm -c
            ${_type_defs} ${_algo_defs} ${EXTRA_DEFS}
            "${KERNEL_SOURCE}"
            -o "${OUTPUT_BC}"
        DEPENDS "${KERNEL_SOURCE}"
        COMMENT
            "[radixsortcl] Compiling ${TYPE_NAME} kernel → SPIR bitcode (clang -target spir64)"
        VERBATIM
    )
endfunction()

# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
function(compile_opencl_kernels)
    cmake_parse_arguments(ARG
        ""
        "TARGET;KERNEL_SOURCE;OUTPUT_DIR"
        "DATA_TYPES;COMPILE_DEFS"
        ${ARGN}
    )

    if(NOT ARG_TARGET)
        message(FATAL_ERROR "compile_opencl_kernels: TARGET is required")
    endif()
    if(NOT ARG_KERNEL_SOURCE)
        message(FATAL_ERROR "compile_opencl_kernels: KERNEL_SOURCE is required")
    endif()
    if(NOT ARG_OUTPUT_DIR)
        set(ARG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernels")
    endif()
    if(NOT ARG_DATA_TYPES)
        set(ARG_DATA_TYPES int32 uint32 int64 uint64)
    endif()

    _rsort_find_tools()

    if(_RSORT_KERNEL_MODE STREQUAL "NONE")
        # No offline compiler available; skip pre-compilation.
        return()
    endif()

    # Map mode → file extension
    if(_RSORT_KERNEL_MODE MATCHES "^SPIRV_")
        set(_ext "spv")
        set(_cmake_mode "SPIRV")
    else()
        set(_ext "bc")
        set(_cmake_mode "SPIR")
    endif()

    file(MAKE_DIRECTORY "${ARG_OUTPUT_DIR}")

    set(_all_outputs "")
    foreach(_type IN LISTS ARG_DATA_TYPES)
        set(_out "${ARG_OUTPUT_DIR}/RadixSort_${_type}.${_ext}")

        if(_RSORT_KERNEL_MODE STREQUAL "SPIRV_CLANG_LLVMSPIRV")
            _rsort_cmd_spirv_clang_llvmspirv(
                "${ARG_KERNEL_SOURCE}" "${_type}" "${_out}" "${ARG_COMPILE_DEFS}")
        elseif(_RSORT_KERNEL_MODE STREQUAL "SPIRV_CLSPV")
            _rsort_cmd_spirv_clspv(
                "${ARG_KERNEL_SOURCE}" "${_type}" "${_out}" "${ARG_COMPILE_DEFS}")
        elseif(_RSORT_KERNEL_MODE STREQUAL "SPIR_CLANG")
            _rsort_cmd_spir_clang(
                "${ARG_KERNEL_SOURCE}" "${_type}" "${_out}" "${ARG_COMPILE_DEFS}")
        endif()

        list(APPEND _all_outputs "${_out}")
    endforeach()

    # Aggregate target so dependants just need to depend on one thing
    set(_agg_target "${ARG_TARGET}_kernels")
    add_custom_target(${_agg_target} DEPENDS ${_all_outputs})
    add_dependencies(${ARG_TARGET} ${_agg_target})

    # Expose output directory and extension to the C++ compiler
    target_compile_definitions(${ARG_TARGET} PRIVATE
        RADIXSORTCL_KERNEL_DIR="${ARG_OUTPUT_DIR}"
        RADIXSORTCL_KERNEL_EXT="${_ext}"
    )

    # Propagate info to the calling scope for install rules
    set(RADIXSORTCL_COMPILED_KERNELS ${_all_outputs} PARENT_SCOPE)
    set(RADIXSORTCL_KERNEL_EXT       ${_ext}         PARENT_SCOPE)
    set(RADIXSORTCL_KERNEL_MODE      ${_cmake_mode}  PARENT_SCOPE)
endfunction()
