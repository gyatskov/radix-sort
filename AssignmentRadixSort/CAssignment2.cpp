/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CAssignment2.h"

#include "CRadixSortTask.h"
#include "CScanTask.h"

#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CAssignment2

bool CAssignment2::DoCompute()
{
	// Task 1: parallel reduction
	cout<<"########################################"<<endl;
	cout<<"Running parallel reduction task..."<<endl<<endl;
	{
		size_t LocalWorkSize[3] = {1, 1, 1};
		CRadixSortTask reduction(1024);
		RunComputeTask(reduction, LocalWorkSize);
	}

	//// Task 2: parallel prefix sum
	//cout<<"########################################"<<endl;
	//cout<<"Running parallel prefix sum task..."<<endl<<endl;
	//{
	//	size_t LocalWorkSize[3] = {256, 1, 1};
	//	CScanTask scan(1024 * 1024 * 64, LocalWorkSize[0]);
	//	RunComputeTask(scan, LocalWorkSize);
	//}


	return true;
}

///////////////////////////////////////////////////////////////////////////////
