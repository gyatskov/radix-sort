#pragma once

#include "../Common/CAssignmentBase.h"

struct RadixSortOptions;

class CRunner : public CAssignmentBase
{
public:
    CRunner(Arguments arguments = Arguments());
	virtual ~CRunner() {};

	//! This overloaded method contains the specific solution of A2
	virtual bool DoCompute();
    
    template <typename DataType>
    void runTask(const RadixSortOptions& options, size_t LocalWorkSize[3]);
};
