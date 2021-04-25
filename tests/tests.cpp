#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "CTestBase.h"

#include "Dataset.h"
#include "RadixSortOptions.h"
#include "CRadixSortTask.h"

#include "../Common/Util.hpp"

template <typename DataType>
auto DatasetCreator(size_t num_elements)
{
    return make_array<std::shared_ptr<Dataset<DataType>>>(
        std::make_shared<Zeros<DataType>>(num_elements),
        std::make_shared<Range<DataType>>(num_elements),
        std::make_shared<InvertedRange<DataType>>(num_elements),
        std::make_shared<RandomDistributed<DataType>>(num_elements),
        std::make_shared<Random<DataType>>(num_elements)
    );
}

class CRunner : public CTestBase
{
public:
    CRunner(Arguments arguments = Arguments());
	virtual ~CRunner() = default;

	bool DoCompute() override;

    template <typename DataType>
    bool runTask(
        const RadixSortOptions& options,
        const std::array<size_t,3>& LocalWorkSize
    );
};

CRunner::CRunner(Arguments arguments /*= Arguments()*/) : CTestBase(arguments)
{ }

template <typename DataType>
bool CRunner::runTask(const RadixSortOptions& options, const std::array<size_t,3>& LocalWorkSize)
{
    const auto datasets = DatasetCreator<DataType>(options.num_elements);
    bool success = true;
    for (const auto& dataset : datasets)
    {
        CRadixSortTask<DataType> radixSort(options, dataset);
        success = success && RunComputeTask(radixSort, LocalWorkSize);
        REQUIRE(success);
    }
    return success;
}

namespace {
template<typename First, typename ...Rest>
bool runAllTypes(CRunner& runner, const RadixSortOptions& options, const std::array<size_t, 3>& localWorkSize)
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

    // LocalWorkSize does not mean anything right here
	const std::array<size_t,3> LocalWorkSize { 1, 1, 1 };
	const auto problemSize = options.num_elements;

    // TODO: Use type list
    return runAllTypes<uint32_t, int32_t, uint64_t, int64_t>(
        *this,
        options,
        LocalWorkSize
    );
}

TEST_CASE( "Main test", "[main]" )
{
    // Non-interactive mode
    constexpr auto argc = 0;
    char* argv[] = {};

    Arguments arguments(argc, argv);
	CRunner radixSortRunner(arguments);

	REQUIRE(radixSortRunner.InitCLContext());

	REQUIRE(radixSortRunner.DoCompute());

	radixSortRunner.ReleaseCLContext();
}
