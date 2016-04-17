/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CRunner.h"

#include "CRadixSortTask.h"
#include "RadixSortOptions.h"

#include <iostream>
#include "Dataset.h"

using namespace std;

CRunner::CRunner(Arguments arguments /*= Arguments()*/) : CAssignmentBase(arguments)
{
}

///////////////////////////////////////////////////////////////////////////////
// CRunner

bool CRunner::DoCompute()
{
    const auto options = RadixSortOptions(m_arguments);

	cout<<"########################################"<<endl;
	cout<<"Running radix sort task..."<<endl<<endl;
	size_t LocalWorkSize[3] = { 1, 1, 1 }; // LocalWorkSize does not mean anything right now
	const auto problemSize = options.num_elements;
	cout << "Sorting " << problemSize << " elements" << std::endl;
	{
		using DataType = uint32_t;
		const std::shared_ptr<Dataset<DataType>> datasets[] = {
			std::make_shared<Zeros<DataType>>(),
			std::make_shared<Range<DataType>>(),
			std::make_shared<InvertedRange<DataType>>(),
			std::make_shared<Random<DataType>>()
		};
		for (const auto dataset : datasets)
		{
			CRadixSortTask<DataType> radixSort(problemSize, dataset);
			RunComputeTask(radixSort, LocalWorkSize);
		}
	}
	{
		using DataType = int32_t;
		const std::shared_ptr<Dataset<DataType>> datasets[] = {
			std::make_shared<Zeros<DataType>>(),
			std::make_shared<Range<DataType>>(),
			std::make_shared<InvertedRange<DataType>>(),
			std::make_shared<Random<DataType>>()
		};
		for (const auto dataset : datasets)
		{
			CRadixSortTask<DataType> radixSort(problemSize, dataset);
			RunComputeTask(radixSort, LocalWorkSize);
		}
	}
	{
		using DataType = uint64_t;
		const std::shared_ptr<Dataset<DataType>> datasets[] = {
			std::make_shared<Zeros<DataType>>(),
			std::make_shared<Range<DataType>>(),
			std::make_shared<InvertedRange<DataType>>(),
			std::make_shared<Random<DataType>>()
		};
		for (const auto dataset : datasets)
		{
			CRadixSortTask<DataType> radixSort(problemSize, dataset);
			RunComputeTask(radixSort, LocalWorkSize);
		}
	}
	{
		using DataType = int64_t;
		const std::shared_ptr<Dataset<DataType>> datasets[] = {
			std::make_shared<Zeros<DataType>>(),
			std::make_shared<Range<DataType>>(),
			std::make_shared<InvertedRange<DataType>>(),
			std::make_shared<Random<DataType>>()
		};
		for (const auto dataset : datasets)
		{
			CRadixSortTask<DataType> radixSort(problemSize, dataset);
			RunComputeTask(radixSort, LocalWorkSize);
		}
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////
