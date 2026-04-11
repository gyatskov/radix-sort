#include "HostData.h"

#include "Dataset.h"

#include <cstdint>
#include <algorithm>

template <typename DataType>
HostDataWithReference<DataType>::HostDataWithReference(std::shared_ptr<Dataset<DataType>> dataset) :
	m_resultSTLCPU(AlgorithmConfiguration::_NUM_MAX_INPUT_ELEMS),
	m_resultRadixSortCPU(AlgorithmConfiguration::_NUM_MAX_INPUT_ELEMS),
    mHostBuffers{ }
{
    {
        mHostBuffers.m_hKeys.resize(AlgorithmConfiguration::_NUM_MAX_INPUT_ELEMS);
        mHostBuffers.m_hHistograms.resize(AlgorithmConfiguration::_RADIX * AlgorithmConfiguration::_NUM_ITEMS);
        mHostBuffers.m_hGlobsum.resize(AlgorithmConfiguration::_NUM_HISTOSPLIT);
        mHostBuffers.h_Permut.resize(AlgorithmConfiguration::_NUM_MAX_INPUT_ELEMS);
        mHostBuffers.h_OutputPermut.resize(AlgorithmConfiguration::_NUM_MAX_INPUT_ELEMS);

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
