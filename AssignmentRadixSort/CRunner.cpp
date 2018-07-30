#include "CRunner.h"

#include "CRadixSortTask.h"
#include "RadixSortOptions.h"

#include <iostream>
#include <array>
#include "Dataset.h"

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
        std::make_shared<Zeros<DataType>>(),
        std::make_shared<Range<DataType>>(),
        std::make_shared<InvertedRange<DataType>>(),
        std::make_shared<RandomDistributed<DataType>>(),
        std::make_shared<Random<DataType>>()
    };
    return result;
}

template <typename DataType>
void CRunner::runTask(const RadixSortOptions& options, size_t LocalWorkSize[3]) 
{
    const auto datasets = DataSetKreator<DataType>(options.num_elements);
    for (const auto dataset : datasets)
    {
        CRadixSortTask<DataType> radixSort(options, dataset);
        RunComputeTask(radixSort, LocalWorkSize);
    }
}

bool CRunner::DoCompute()
{
    const auto options = RadixSortOptions(m_arguments);

    using allowedTypes = std::tuple<uint32_t, int32_t, uint64_t, int64_t>;

	cout<<"########################################"<<endl;
	cout<<"Running radix sort task..."<<endl<<endl;
	size_t LocalWorkSize[3] = { 1, 1, 1 }; // LocalWorkSize does not mean anything right now
	const auto problemSize = options.num_elements;
	cout << "Sorting " << problemSize << " elements" << std::endl;
    runTask<uint32_t>(options, LocalWorkSize);
    runTask<int32_t> (options, LocalWorkSize);
    runTask<uint64_t>(options, LocalWorkSize);
    runTask<int64_t> (options, LocalWorkSize);

	return true;
}

///////////////////////////////////////////////////////////////////////////////
