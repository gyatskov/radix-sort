#include "HostData.h"

#include "Dataset.h"

#include <cstdint>
#include <algorithm>

template <typename DataType>
HostDataWithReference<DataType>::HostDataWithReference(std::shared_ptr<Dataset<DataType>> dataset) :
	m_resultSTLCPU(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_resultRadixSortCPU(Parameters::_NUM_MAX_INPUT_ELEMS),
    mHostBuffers{ }
{
    {
        mHostBuffers.m_hKeys.resize(Parameters::_NUM_MAX_INPUT_ELEMS);
        mHostBuffers.m_hHistograms.resize(Parameters::_RADIX * Parameters::_NUM_ITEMS);
        mHostBuffers.m_hGlobsum.resize(Parameters::_NUM_HISTOSPLIT);
        mHostBuffers.h_Permut.resize(Parameters::_NUM_MAX_INPUT_ELEMS);

        std::iota(mHostBuffers.h_Permut.begin(), mHostBuffers.h_Permut.end(), 0);
    }

	std::copy(
        dataset->dataset.begin(),
        dataset->dataset.end(),
        mHostBuffers.m_hKeys.begin()
    );
}

// Specialize datasets for exactly these four types.
template struct HostDataWithReference < int32_t > ;
template struct HostDataWithReference < int64_t > ;
template struct HostDataWithReference < uint32_t > ;
template struct HostDataWithReference < uint64_t > ;
