#include "CRunner.h"

#include <catch2/catch.hpp>

#include "CRadixSortTask.h"
#include "RadixSortOptions.h"
#include "Dataset.h"

#include <array>
#include <memory>

using namespace std;

CRunner::CRunner(Arguments arguments /*= Arguments()*/) : CAssignmentBase(arguments)
{
}

///////////////////////////////////////////////////////////////////////////////
// CRunner
static const size_t NUM_DATASETS = 5;

/// [sic]
template <typename DataType>
std::array<std::shared_ptr<Dataset<DataType>>, NUM_DATASETS> DataSetKreator(size_t num_elements)
{
    std::array<std::shared_ptr<Dataset<DataType>>, NUM_DATASETS> result = {
        std::make_shared<Zeros<DataType>>(num_elements),
        std::make_shared<Range<DataType>>(num_elements),
        std::make_shared<InvertedRange<DataType>>(num_elements),
        std::make_shared<RandomDistributed<DataType>>(num_elements),
        std::make_shared<Random<DataType>>(num_elements)
    };
    return result;
}

template <typename DataType>
bool CRunner::runTask(const RadixSortOptions& options, size_t LocalWorkSize[3])
{
    const auto datasets = DataSetKreator<DataType>(options.num_elements);
    bool success = true;
    for (const auto dataset : datasets)
    {
        CRadixSortTask<DataType> radixSort(options, dataset);
        success = success && RunComputeTask(radixSort, LocalWorkSize);
        REQUIRE(success);
    }
    return success;
}

namespace {
template<typename First, typename ...Rest>
bool runAllTypes(CRunner& runner, const RadixSortOptions& options, size_t localWorkSize[3])
{
    bool success = runner.runTask<First>(options, localWorkSize);

    if constexpr(sizeof...(Rest) > 0) {
        success = success && runAllTypes<Rest...>(runner, options, localWorkSize);
    }
    return success;
}
} // namespace

bool CRunner::DoCompute()
{
    const auto options = RadixSortOptions(m_arguments);

	size_t LocalWorkSize[3] = { 1, 1, 1 }; // LocalWorkSize does not mean anything right now
	const auto problemSize = options.num_elements;

    return runAllTypes<uint32_t, int32_t, uint64_t, int64_t>(*this, options, LocalWorkSize);
}

///////////////////////////////////////////////////////////////////////////////
