#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include "CTestBase.h"

#include "Dataset.h"
#include "RadixSortOptions.h"
#include "CRadixSortTask.h"
#include <exception>
#include <cstdlib> // getenv

#include "Common/Util.hpp"
// TODO: Move
#include <CL/Utils/Error.hpp>

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
    CRunner() = default;
	virtual ~CRunner() = default;

	bool DoCompute(std::vector<std::string> arguments = {}) override;

    template <typename DataType>
    bool runTask(
        const RadixSortOptions& options,
        const LocalWorkSize& localWorkSize
    );
};

template <typename DataType>
bool CRunner::runTask(const RadixSortOptions& options, const LocalWorkSize& localWorkSize)
{
    const auto datasets = DatasetCreator<DataType>(options.num_elements);
    bool success = true;
    for (const auto& dataset : datasets)
    {
        CRadixSortTask<DataType> radixSort(options, dataset);
        success = success && RunComputeTask(radixSort, localWorkSize);
        REQUIRE(success);
    }
    return success;
}

namespace {
template<typename First, typename ...Rest>
bool runAllTypes(CRunner& runner, const RadixSortOptions& options, const LocalWorkSize& localWorkSize)
{
    bool success = runner.runTask<First>(options, localWorkSize);

    if constexpr(sizeof...(Rest) > 0) {
        success = success && runAllTypes<Rest...>(runner, options, localWorkSize);
    }
    return success;
}
} // namespace

bool CRunner::DoCompute(std::vector<std::string> arguments /*= {}*/)
{
    const auto options = ParseArgs(arguments);

    // LocalWorkSize does not mean anything right here
	const LocalWorkSize localWorkSize { 1, 1, 1 };

    // TODO: Use type list
    return runAllTypes<uint32_t, int32_t, uint64_t, int64_t>(
        *this,
        options,
        localWorkSize
    );
}

TEST_CASE( "Main test", "[main]" )
{
    // Non-interactive mode
    const auto numElements = [&]{
        if(const char* numElements = std::getenv("RADIXSORT_INPUT_ELEMENTS"))
        {
            return std::string(numElements);

        } else {
            return std::to_string(AlgorithmConfiguration::_NUM_MAX_INPUT_ELEMS);
        }
    }();

    const auto numIterations = [&]{
        if(const char* numIterations = std::getenv("RADIXSORT_ITERATIONS"))
        {
            return std::string(numIterations);

        } else {
            return std::to_string(1U);
        }
    }();


	CRunner radixSortRunner;

    try {
        const auto initialized = radixSortRunner.InitCLContext();
        REQUIRE(initialized);
        const auto status = radixSortRunner.DoCompute(
            {
                "--num-elements", numElements,
                "--num-iterations", numIterations,
            }
        );
        REQUIRE(status == true);
    } catch(const cl::Error& exc) {
        INFO("CL Error: " << std::string(exc.what()));
        INFO(exc.err() << "(" << std::hex << exc.err() << ")");
        REQUIRE(false);
    } catch(const cl::util::Error& exc) {
        const auto str = std::string(exc.what());
        INFO("CL Util Error: " << str);
        INFO(exc.err() << "(" << std::hex << exc.err() << ")");
        REQUIRE(false);
    } catch(const std::exception& exc) {
        INFO("Unhandled: " << std::string(exc.what()));
        REQUIRE(false);
    }
}
