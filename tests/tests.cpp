#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include "Dataset.h"
#include "RadixSortOptions.h"
#include "CRadixSortTask.h"

#include "Common/Util.hpp"
#include "Common/ComputeState.h"
// TODO: Move
#include <CL/Utils/Error.hpp>

#include <exception>
#include <print>
#include <cstdlib> // getenv

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

namespace {


template<typename First, typename ...Rest>
bool runAllTypes(auto& runner, const RadixSortOptions& options, const LocalWorkSize& localWorkSize)
{
    bool success = runner.template runTask<First>(options, localWorkSize);

    if constexpr(sizeof...(Rest) > 0) {
        success = success && runAllTypes<Rest...>(runner, options, localWorkSize);
    }
    return success;
}


template<class Task>
concept ComputeTask = requires(Task t, cl::Device dev, cl::Context ctx){
    {t.InitResources(dev, ctx)} -> std::same_as<bool>;
    {t.ReleaseResources()} -> std::same_as<void>;
};

class CRunner 
{
public:
    CRunner() = default;
	virtual ~CRunner() = default;

    bool DoCompute(std::vector<std::string> arguments /*= {}*/)
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

    template <typename DataType>
    bool runTask(const RadixSortOptions& options, const LocalWorkSize& localWorkSize)
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

	bool InitCLContext() {
        return m_computeState.init();
    }

    bool RunComputeTask(ComputeTask auto& task, const LocalWorkSize& localWorkSize)
    {
        if(m_computeState.m_CLContext() == nullptr)
        {
            std::println(std::cerr, "Error: RunComputeTask() cannot execute because the OpenCL context is null.");
            return false;
        }

        if(!task.InitResources(
                m_computeState.device(),
                m_computeState.m_CLContext
            )
        )
        {
            std::println(std::cerr, "Error during resource allocation. Aborting execution.");
            task.ReleaseResources();
            return false;
        }

        // Compute the golden result.
        std::println("Computing CPU reference result...");
        task.ComputeCPU();
        std::println("DONE");

        // Running the same task on the GPU.
        std::println("Computing GPU result...");

        // Runing the kernel N times. This make the measurement of the execution time more accurate.
        task.ComputeGPU(
                m_computeState.m_CLContext,
                m_computeState.m_CLCommandQueue,
                localWorkSize
        );
        std::println("DONE");

        // Validating results.
        std::string result = "GOLD TEST PASSED!\n";
        if (!task.ValidateResults())
        {
            result = "INVALID RESULTS!\n";
        }
        std::println("{}", task.ValidateResults() ? "GOLD TEST PASSED!" : "INVALID RESULTS");

        // Cleaning up.
        task.ReleaseResources();

        return true;
    }
private:
    ComputeState m_computeState;
};

} // namespace


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
