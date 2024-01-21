#pragma once

#include "../Common/IComputeTask.h"
#include "Parameters.h"
#include "HostData.h"
#include "RadixSortGPU.h"
#include "RadixSortOptions.h"
#include "OperationStatus.h"
#include "Statistics.h"

#include <map>
#include <memory>
#include <cstdint>
#include <array>

/// Runtime statistics of CPU implementation algorithms
struct RuntimesCPU {
    Statistics timeRadix{};
    Statistics timeSTL{};
};

/// Parallel radix sort orchestrator
/// @tparam T Type of data to be sorted
/// @TODO: Turn into a test class
template <typename T>
class CRadixSortTask : public IComputeTask
{
public:
	using DataType = T;

    CRadixSortTask(
        const RadixSortOptions& options,
        std::shared_ptr<Dataset<DataType>> dataset
    );

	virtual ~CRadixSortTask() = default;

    ///////////////////////////////////////////////////////////////
	// IComputeTask realization
	bool InitResources(cl::Device Device, cl::Context Context) override;
	void ReleaseResources() override;
	void ComputeGPU(
        cl::Context Context,
        cl::CommandQueue CommandQueue,
        const std::array<size_t,3>& LocalWorkSize
    ) override;

    void ComputeCPU() override;

    /** Tests results validity **/
	bool ValidateResults() override;
    ///////////////////////////////////////////////////////////////

protected:
    using Parameters = AlgorithmParameters<DataType>;

	// Helper methods
	void CheckLocalMemory(cl::Device Device);
	uint32_t Resize(uint32_t nn);

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
        const std::array<size_t,3>& LocalWorkSize
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
    RadixSortGPU<DataType> mRadixSortGPU;
    /// Options provided by user
    RadixSortOptions mOptions;
};

/** Measures task performance **/
template<class Callable>
void TestPerformance(
    cl::CommandQueue CommandQueue,
    Callable&& fun,
    const RadixSortOptions& options,
    const size_t numIterations,
    size_t numberKeys,
    const std::string& datasetName,
    const std::string& datatype
);

/** Writes performance to stream **/
template <typename Stream>
void writePerformance(
    Stream&& stream,
    const RuntimesGPU& runtimesGPU,
    const RuntimesCPU& runtimesCPU,
    size_t numberKeys,
    const std::string& datasetName,
    const std::string& datatype
);

