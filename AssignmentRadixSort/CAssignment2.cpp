/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CAssignment2.h"

#include "CRadixSortTask.h"

#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CAssignment2

bool CAssignment2::DoCompute()
{
	cout<<"########################################"<<endl;
	cout<<"Running radix sort task..."<<endl<<endl;
	{
		using DataType = uint32_t;
		size_t LocalWorkSize[3] = {1, 1, 1}; // LocalWorkSize does not mean anything right now
        const auto problemSize = 1 << 15;
		CRadixSortTask<DataType> radixSort(problemSize);
        RunComputeTask(radixSort, LocalWorkSize);
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////
