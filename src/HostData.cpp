#include "HostData.h"

#include "Dataset.h"

#include <cstdint>
#include <algorithm>

template <typename DataType>
HostBuffers<DataType>::HostBuffers() :
	m_hKeys(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_hHistograms(Parameters::_RADIX * Parameters::_NUM_ITEMS),
	m_hGlobsum(Parameters::_NUM_HISTOSPLIT),
	h_Permut(Parameters::_NUM_MAX_INPUT_ELEMS)
{
	std::iota(h_Permut.begin(), h_Permut.end(), 0);
}

template <typename DataType>
HostData<DataType>::HostData(std::shared_ptr<Dataset<DataType>> dataset) :
	m_resultSTLCPU(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_resultRadixSortCPU(Parameters::_NUM_MAX_INPUT_ELEMS),
    mHostBuffers{std::make_shared<DataBuffers>()}
{
	std::copy(
        dataset->dataset.begin(),
        dataset->dataset.end(),
        mHostBuffers->m_hKeys.begin());
}

// Specialize datasets for exactly these four types.
template struct HostData < int32_t > ;
template struct HostData < int64_t > ;
template struct HostData < uint32_t > ;
template struct HostData < uint64_t > ;
