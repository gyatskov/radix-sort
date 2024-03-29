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
    LOADING_SOURCE_FAILED,
};
