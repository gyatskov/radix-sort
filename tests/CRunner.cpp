#include "CRunner.h"

#include <catch2/catch.hpp>

#include "CRadixSortTask.h"
#include "RadixSortOptions.h"
#include "Dataset.h"

#include <array>
#include <memory>
#include <type_traits>

namespace {

template <typename Dest=void, typename ...Arg>
constexpr auto make_array(Arg&& ...arg) {
   if constexpr (std::is_same<void,Dest>::value)
      return std::array<std::common_type_t<std::decay_t<Arg>...>, sizeof...(Arg)>{{ std::forward<Arg>(arg)... }};
   else
      return std::array<Dest, sizeof...(Arg)>{{ std::forward<Arg>(arg)... }};
}
} // namespace

CRunner::CRunner(Arguments arguments /*= Arguments()*/) : CAssignmentBase(arguments)
{
}

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

template <typename DataType>
bool CRunner::runTask(const RadixSortOptions& options, const std::array<size_t,3>& LocalWorkSize)
{
    const auto datasets = DatasetCreator<DataType>(options.num_elements);
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

    return runAllTypes<uint32_t, int32_t, uint64_t, int64_t>(*this, options, LocalWorkSize);
}

///////////////////////////////////////////////////////////////////////////////
