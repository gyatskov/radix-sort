#include "HostData.h"


HostData::HostData() :
	m_hKeys(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_hCheckKeys(Parameters::_NUM_MAX_INPUT_ELEMS),
	h_Permut(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_hHistograms(Parameters::_RADIX * Parameters::_NUM_GROUPS * Parameters::_NUM_ITEMS_PER_GROUP),
	m_hGlobsum(Parameters::_NUM_HISTOSPLIT),
	m_resultSTLCPU(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_resultRadixSortCPU(Parameters::_NUM_MAX_INPUT_ELEMS),

	m_distributedRandom(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_zeros(Parameters::_NUM_MAX_INPUT_ELEMS),
	m_invertedRange(Parameters::_NUM_MAX_INPUT_ELEMS)
{
	std::iota(h_Permut.begin(), h_Permut.end(), 0);

	std::fill(m_zeros.begin(), m_zeros.end(), 0);
	std::iota(m_invertedRange.begin(), m_invertedRange.end(), 0);
	std::reverse(m_invertedRange.begin(), m_invertedRange.end());

	std::string seedStr("Schmutz :P");
	std::seed_seq seed(seedStr.begin(), seedStr.end());
	std::mt19937 generator(seed);

	/// TODO("Adapt to signed integers, later on.");
	std::uniform_int_distribution<DataType> dis(0, std::numeric_limits<DataType>::max());
	// fill the array with some values
	std::generate(m_distributedRandom.begin(), m_distributedRandom.end(), std::bind(dis, generator));

	const auto& sequenceToBeSorted = m_invertedRange;

	std::copy(sequenceToBeSorted.begin(), sequenceToBeSorted.end(), m_hKeys.begin());
	std::copy(m_hKeys.begin(), m_hKeys.end(), m_hCheckKeys.begin());
}