#pragma once

/// @note Radix sort specific
enum class OperationStatus {
    OK,
    HOST_BUFFERS_FAILED,
    INITIALIZATION_FAILED,
    DATA_UPLOAD_FAILED,
    CALCULATION_FAILED,
    DATA_DOWNLOAD_FAILED,
    CLEANUP_FAILED,
    RESIZE_FAILED,
    KERNEL_CREATION_FAILED,
    PROGRAM_CREATION_FAILED,
    NO_SOURCE_FOUND,
    LOADING_SOURCE_FAILED,
    /// Pre-compiled SPIR-V binary (.spv) could not be read from disk
    LOADING_SPIRV_FAILED,
    /// Pre-compiled SPIR bitcode (.bc) could not be read from disk
    LOADING_BINARY_FAILED,
    /// clCreateProgramWithILKHR (SPIR-V via cl_khr_il_program) failed
    PROGRAM_IL_CREATION_FAILED,
    /// clCreateProgramWithBinary (SPIR via cl_khr_spir) failed
    PROGRAM_BINARY_CREATION_FAILED,
};
