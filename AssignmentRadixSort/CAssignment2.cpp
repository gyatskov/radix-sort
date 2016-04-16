/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CAssignment2.h"

#include "CRadixSortTask.h"

#include <iostream>
#include "Dataset.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CAssignment2

bool CAssignment2::DoCompute()
{
	cout<<"########################################"<<endl;
	cout<<"Running radix sort task..."<<endl<<endl;
	size_t LocalWorkSize[3] = { 1, 1, 1 }; // LocalWorkSize does not mean anything right now
	const auto problemSize = 1 << 10;
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
