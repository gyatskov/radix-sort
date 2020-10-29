#pragma once

#include "../Common/CAssignmentBase.h"

#include <array>

struct RadixSortOptions;

class CRunner : public CAssignmentBase
{
public:
    CRunner(Arguments arguments = Arguments());
	virtual ~CRunner() {};

	virtual bool DoCompute();

    template <typename DataType>
    bool runTask(const RadixSortOptions& options, const std::array<size_t,3>& LocalWorkSize);
};
