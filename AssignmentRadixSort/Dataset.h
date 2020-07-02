#pragma once

#include "Parameters.h"

#include <vector>
#include <map>
#include <string>
#include <functional>
#include <cstdint>
#include <random>
#include <algorithm>
#include <numeric>


template <typename DataType>
using Parameters = AlgorithmParameters<DataType>;

template <typename _DataType>
struct Dataset
{
	using DataType = _DataType;

	static const char* const name;
	std::vector<DataType> dataset;

    Dataset(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS) : dataset(size)
	{}

	virtual const char* const getName() {
		return name;
	}
};


template <typename DataType>
struct Zeros : Dataset<DataType>
{
	static const char* const name;

    Zeros(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS);

	virtual const char* const getName() {
		return Zeros::name;
	}
};

template <typename DataType>
struct RandomDistributed : Dataset < DataType >
{
	static const char* const name;

    RandomDistributed(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS);

	virtual const char* const getName() {
		return RandomDistributed::name;
	}
};

template <typename DataType>
struct Random : Dataset < DataType >
{
    static const char* const name;

    Random(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS);

    virtual const char* const getName() {
        return Random::name;
    }
};

template <typename DataType>
struct Range : Dataset < DataType >
{
	static const char* const name;

    Range(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS);

	virtual const char* const getName() {
		return Range::name;
	}
};

template <typename DataType>
struct InvertedRange : Dataset < DataType >
{
	static const char* const name;

    InvertedRange(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS);

	virtual const char* const getName() {
		return InvertedRange::name;
	}
};

// template <typename DataType>
// const char* const Zeros<DataType>::getName()
// {
// 	return name;
// }
// template <typename DataType>
// const char* const Random<DataType>::getName()
// {
// 	return name;
// }
// template <typename DataType>
// const char* const Range<DataType>::getName()
// {
// 	return name;
// }
// template <typename DataType>
// const char* const InvertedRange<DataType>::getName()
// {
// 	return name;
// }

template <typename DataType>
const char* const Dataset<DataType>::name = "UNKNOWN";

template <typename DataType>
const char* const Zeros<DataType>::name = "Zeros";

template <typename DataType>
const char* const RandomDistributed<DataType>::name = "Uniform random";

template <typename DataType>
const char* const Random<DataType>::name = "Random";

template <typename DataType>
const char* const Range<DataType>::name = "Range";

template <typename DataType>
const char* const InvertedRange<DataType>::name = "Inverted range";

template <typename DataType>
Zeros<DataType>::Zeros(std::size_t size)
    : Dataset<DataType>(size)
{
    auto& dataset = Dataset<DataType>::dataset;
	std::fill(dataset.begin(), dataset.end(), 0);
}

template <typename DataType>
RandomDistributed<DataType>::RandomDistributed(std::size_t size)
    : Dataset<DataType>(size)
{
	std::string seedStr("Test :P");
	std::seed_seq seed(seedStr.begin(), seedStr.end());
	std::mt19937 generator(seed);

	std::uniform_int_distribution<DataType> dis(std::numeric_limits<DataType>::min(), std::numeric_limits<DataType>::max());
	// fill the array with some values
    auto& dataset = Dataset<DataType>::dataset;
	std::generate(dataset.begin(), dataset.end(), std::bind(dis, generator));

	// Ensure that min and max are in the input array
	*dataset.begin() = std::numeric_limits<DataType>::max();
	*(dataset.end() - 1) = std::numeric_limits<DataType>::min();
}

template <typename DataType>
Random<DataType>::Random(std::size_t size)
    : Dataset<DataType>(size)
{
    std::string seedStr("Random test string");
    std::seed_seq seed(seedStr.begin(), seedStr.end());
    std::mt19937 generator(seed);

    // fill the array with some values
    auto& dataset = Dataset<DataType>::dataset;
    std::generate(dataset.begin(), dataset.end(), generator);
}

template <typename DataType>
InvertedRange<DataType>::InvertedRange(std::size_t size)
    : Dataset<DataType>(size)
{
    auto& dataset = Dataset<DataType>::dataset;
	std::iota(dataset.begin(), dataset.end(), std::numeric_limits<DataType>::min());
	std::reverse(dataset.begin(), dataset.end());
}

template <typename DataType>
Range<DataType>::Range(std::size_t size)
    : Dataset<DataType>(size)
{
    auto& dataset = Dataset<DataType>::dataset;
	std::iota(dataset.begin(), dataset.end(), std::numeric_limits<DataType>::min());
}

// Specialize datasets for exactly these four types.
template struct Dataset < int32_t > ;
template struct Dataset < int64_t > ;
template struct Dataset < uint32_t > ;
template struct Dataset < uint64_t > ;

template struct RandomDistributed < int32_t > ;
template struct RandomDistributed < int64_t > ;
template struct RandomDistributed < uint32_t > ;
template struct RandomDistributed < uint64_t > ;

template struct Random < int32_t >;
template struct Random < int64_t >;
template struct Random < uint32_t >;
template struct Random < uint64_t >;


template struct Zeros < int32_t > ;
template struct Zeros < int64_t > ;
template struct Zeros < uint32_t > ;
template struct Zeros < uint64_t > ;

template struct Range < int32_t > ;
template struct Range < int64_t > ;
template struct Range < uint32_t > ;
template struct Range < uint64_t > ;

template struct InvertedRange < int32_t >;
template struct InvertedRange < int64_t >;
template struct InvertedRange < uint32_t >;
template struct InvertedRange < uint64_t >;
