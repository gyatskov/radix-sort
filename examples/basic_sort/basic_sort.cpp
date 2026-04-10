/// @file basic_sort.cpp
/// @brief Minimal example showing how to sort integer data on the GPU
///        using the radixsortcl library.
///
/// Build (from project root):
///   cmake -B build && cmake --build build
///
/// Run:
///   ./build/examples/basic_sort

#include "Common/ComputeState.h"      // OpenCL platform/device/context/queue setup
#include "RadixSortGPU.h"      // GPU radix sort algorithm
#include "Dataset.h"           // Built-in dataset generators

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

/// Sort `numElements` random uint32_t values on the GPU and verify the result.
template <typename DataType>
bool sortAndVerify(ComputeState& compute, uint32_t numElements)
{
    // ------------------------------------------------------------------
    // 1. Create a dataset with random data
    // ------------------------------------------------------------------
    RandomDistributed<DataType> dataset(numElements);

    // ------------------------------------------------------------------
    // 2. Sort on the GPU with a single call
    // ------------------------------------------------------------------
    RadixSortGPU<DataType> sorter;
    sorter.setLogStream(&std::cout);

    std::vector<DataType> result;
    auto status = sorter.sort(
        compute.device(),
        compute.m_CLContext,
        compute.m_CLCommandQueue,
        std::span<const DataType>(dataset.dataset.data(), numElements),
        result
    );
    if (status != OperationStatus::OK) {
        std::cerr << "GPU sort failed\n";
        return false;
    }

    // ------------------------------------------------------------------
    // 3. Verify against std::sort
    // ------------------------------------------------------------------
    std::vector<DataType> reference(dataset.dataset.begin(),
                                    dataset.dataset.begin() + numElements);
    std::sort(reference.begin(), reference.end());

    const bool correct = std::equal(
        reference.begin(), reference.end(),
        result.begin()
    );

    // ------------------------------------------------------------------
    // 4. Print timing information
    // ------------------------------------------------------------------
    const auto runtimes = sorter.getRuntimes();
    std::cout << "\n--- Timing (avg ms) ---\n"
              << "  Histogram : " << runtimes.timeHisto.avg  << "\n"
              << "  Scan      : " << runtimes.timeScan.avg   << "\n"
              << "  Reorder   : " << runtimes.timeReorder.avg << "\n"
              << "  Paste     : " << runtimes.timePaste.avg  << "\n"
              << "  Total     : " << runtimes.timeTotal.avg  << "\n";

    return correct;
}

int main()
{
    // ------------------------------------------------------------------
    // Set up OpenCL (platform, device, context, command queue)
    // ------------------------------------------------------------------
    ComputeState compute;
    if (!compute.init()) {
        std::cerr << "No suitable OpenCL GPU device found.\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // Sort 1 048 576 random uint32_t values
    // ------------------------------------------------------------------
    constexpr uint32_t N = 1U << 20U;  // ~1M elements

    std::cout << "Sorting " << N << " uint32_t values on the GPU...\n\n";
    const bool ok = sortAndVerify<uint32_t>(compute, N);

    std::cout << "\nResult: " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok ? 0 : 1;
}
