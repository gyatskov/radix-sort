#pragma once

#include "Parameters.h"
#include "HostData.h"
#include "RadixSortGPU.h"
#include "RadixSortOptions.h"
#include "Statistics.h"

#include "Common/CommonDefs.h"

#include <memory>
#include <cstdint>

/// Runtime statistics of CPU implementation algorithms
struct RuntimesCPU {
    Statistics timeRadix{};
    Statistics timeSTL{};
};

/// Parallel radix sort orchestrator
/// @tparam T Type of data to be sorted
/// @TODO: Turn into a test class
template <typename T>
class CRadixSortTask 
{
public:
	using DataType = T;

    CRadixSortTask(
        const RadixSortOptions& options,
        std::shared_ptr<Dataset<DataType>> dataset
    );

	virtual ~CRadixSortTask() = default;

    ///////////////////////////////////////////////////////////////
	bool InitResources(cl::Device Device, cl::Context Context);
	void ReleaseResources();
	void ComputeGPU(
        cl::Context Context,
        cl::CommandQueue CommandQueue,
        const LocalWorkSize& LocalWorkSize
    );

    void ComputeCPU();

    /** Tests results validity **/
	bool ValidateResults();
    ///////////////////////////////////////////////////////////////

protected:
    using Parameters = AlgorithmParameters<DataType>;

	// Helper methods
	void CheckLocalMemory(cl::Device Device);

    /// Performs reorder step
	void Reorder(
        cl::CommandQueue CommandQueue,
        int pass
    );
    ///////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////

	void ExecuteTask(
        cl::Context Context,
        cl::CommandQueue CommandQueue,
        const LocalWorkSize& localWorkSize
    );


    uint32_t mNumberKeys{0U}; // actual number of keys
    uint32_t mNumberKeysRounded{0U}; // next multiple of _ITEMS*_GROUPS

    // Actual host data:
    // * intermediate algorithm buffers
    // * reference results
    HostDataWithReference<DataType> mHostData;

	// data set used for tests
    using TypedDataset = Dataset<DataType>;
	std::shared_ptr<TypedDataset> m_selectedDataset;

	// Runtime statistics CPU
    RuntimesCPU mRuntimesCPU{};

    /// Main GPU Radix Sort algorithm
    RadixSortGPU mRadixSortGPU;
    /// Options provided by user
    RadixSortOptions mOptions;
};
