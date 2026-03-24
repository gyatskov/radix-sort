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
#include "Parameters.h"        // Algorithm compile-time parameters

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

/// Sort `numElements` random uint32_t values on the GPU and verify the result.
template <typename DataType>
bool sortAndVerify(ComputeState& compute, uint32_t numElements)
{
    using Parameters = AlgorithmParameters<DataType>;

    // ------------------------------------------------------------------
    // 1. Create a dataset with random data
    // ------------------------------------------------------------------
    RandomDistributed<DataType> dataset(numElements);

    // ------------------------------------------------------------------
    // 2. Allocate host buffers
    //    The algorithm needs several auxiliary buffers in addition to the
    //    input/output key arrays.
    // ------------------------------------------------------------------
    RadixSortGPU<DataType> sorter;

    const uint32_t numRounded = sorter.Resize(numElements);

    std::vector<DataType>  hKeys(numRounded);
    std::vector<DataType>  hResult(numRounded);
    std::vector<uint32_t>  hHistograms(Parameters::_RADIX * Parameters::_NUM_ITEMS);
    std::vector<uint32_t>  hGlobsum(Parameters::_NUM_HISTOSPLIT);
    std::vector<uint32_t>  hPermut(numRounded);

    // Copy the dataset into the key buffer
    std::copy_n(dataset.dataset.begin(), numElements, hKeys.begin());

    // Initialize the permutation to the identity
    std::iota(hPermut.begin(), hPermut.end(), 0U);

    // Build non-owning spans that the sorter will reference
    HostSpans<DataType> spans {
        { hKeys.data(),       hKeys.size()       },
        { hHistograms.data(), hHistograms.size()  },
        { hGlobsum.data(),    hGlobsum.size()     },
        { hPermut.data(),     hPermut.size()      },
        { hResult.data(),     hResult.size()      },
    };

    // ------------------------------------------------------------------
    // 3. Initialize the GPU sorter (compiles OpenCL kernels, allocates
    //    device buffers)
    // ------------------------------------------------------------------
    auto status = sorter.initialize(
        compute.device(),
        compute.m_CLContext,
        numElements,
        spans
    );
    if (status != OperationStatus::OK) {
        std::cerr << "Failed to initialize RadixSortGPU\n";
        return false;
    }

    // Optional: enable diagnostic output
    sorter.setLogStream(&std::cout);

    // ------------------------------------------------------------------
    // 4. Upload -> Sort -> Download
    // ------------------------------------------------------------------
    auto& queue = compute.m_CLCommandQueue;

    // Pad any extra elements beyond numElements with large values so they
    // sort to the end and don't interfere with the real data.
    if (numRounded != numElements) {
        sorter.padGPUData(queue, sizeof(DataType) * numElements);
    }

    status = sorter.uploadData(queue);
    if (status != OperationStatus::OK) {
        std::cerr << "Upload failed\n";
        return false;
    }

    status = sorter.calculate(queue);
    if (status != OperationStatus::OK) {
        std::cerr << "GPU sort failed\n";
        return false;
    }

    status = sorter.downloadData(queue);
    if (status != OperationStatus::OK) {
        std::cerr << "Download failed\n";
        return false;
    }

    // ------------------------------------------------------------------
    // 5. Verify against std::sort
    // ------------------------------------------------------------------
    std::vector<DataType> reference(dataset.dataset.begin(),
                                    dataset.dataset.begin() + numElements);
    std::sort(reference.begin(), reference.end());

    const bool correct = std::equal(
        reference.begin(), reference.end(),
        hResult.begin()
    );

    // ------------------------------------------------------------------
    // 6. Print timing information
    // ------------------------------------------------------------------
    const auto runtimes = sorter.getRuntimes();
    std::cout << "\n--- Timing (avg ms) ---\n"
              << "  Histogram : " << runtimes.timeHisto.avg  << "\n"
              << "  Scan      : " << runtimes.timeScan.avg   << "\n"
              << "  Reorder   : " << runtimes.timeReorder.avg << "\n"
              << "  Paste     : " << runtimes.timePaste.avg  << "\n"
              << "  Total     : " << runtimes.timeTotal.avg  << "\n";

    // ------------------------------------------------------------------
    // 7. Cleanup
    // ------------------------------------------------------------------
    sorter.release();

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
