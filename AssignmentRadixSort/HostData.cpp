#include "HostData.h"

#include <cstdint>

template <typename DataType>
HostData<DataType>::HostData(std::shared_ptr<Dataset<DataType>> dataset) :
	m_hKeys(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_hCheckKeys(Parameters::_NUM_MAX_INPUT_ELEMS),
	h_Permut(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_hHistograms(Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP),
	m_hGlobsum(Parameters::_NUM_HISTOSPLIT),
	m_resultSTLCPU(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_resultRadixSortCPU(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_selectedDataset(dataset)
{
	std::iota(h_Permut.begin(), h_Permut.end(), 0);

	std::copy(m_selectedDataset->dataset.begin(), m_selectedDataset->dataset.end(), m_hKeys.begin());
	std::copy(m_hKeys.begin(), m_hKeys.end(), m_hCheckKeys.begin());
}

template struct HostData < int32_t > ;
template struct HostData < int64_t > ;
template struct HostData < uint32_t > ;
template struct HostData < uint64_t > ;