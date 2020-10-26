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

	virtual const char* name() const {return "UNKNOWN";}

	std::vector<DataType> dataset;

    Dataset(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS) : dataset(size)
	{}

    virtual ~Dataset() = default;
};


template <typename DataType>
struct Zeros : Dataset<DataType>
{
	virtual const char* name() const override {return "Zeros";}

    Zeros(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS);
    virtual ~Zeros() = default;
};

template <typename DataType>
struct RandomDistributed : Dataset < DataType >
{
	virtual const char* name() const override {return "Random Uniform";}

    RandomDistributed(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS);
    virtual ~RandomDistributed() = default;
};

template <typename DataType>
struct Random : Dataset < DataType >
{
	virtual const char* name() const override {return "Random Random";}

    Random(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS);
    virtual ~Random() = default;
};

template <typename DataType>
struct Range : Dataset < DataType >
{
	virtual const char* name() const override {return "Range";}

    Range(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS);
    virtual ~Range() = default;
};

template <typename DataType>
struct InvertedRange : Dataset < DataType >
{
	virtual const char* name() const override {return "Inverted Range";}

    InvertedRange(std::size_t size = Parameters<DataType>::_NUM_MAX_INPUT_ELEMS);
    virtual ~InvertedRange() = default;
};


template <typename DataType>
Zeros<DataType>::Zeros(std::size_t size)
    : Dataset<DataType>(size)
{
    auto& ds = Dataset<DataType>::dataset;
	std::fill(ds.begin(), ds.end(), 0);
}

template <typename DataType>
RandomDistributed<DataType>::RandomDistributed(std::size_t size)
    : Dataset<DataType>(size)
{
	const std::string seedStr("Random Test Seed");
	std::seed_seq seed(seedStr.begin(), seedStr.end());
	std::mt19937 generator(seed);

	std::uniform_int_distribution<DataType> dis(std::numeric_limits<DataType>::min(), std::numeric_limits<DataType>::max());
	// fill the array with some values
    auto& ds = Dataset<DataType>::dataset;
	std::generate(ds.begin(), ds.end(), std::bind(dis, generator));

	// Ensure that min and max are in the input array
	*ds.begin() = std::numeric_limits<DataType>::max();
	*(ds.end() - 1) = std::numeric_limits<DataType>::min();
}

template <typename DataType>
Random<DataType>::Random(std::size_t size)
    : Dataset<DataType>(size)
{
	const std::string seedStr("Random Test Seed");
    std::seed_seq seed(seedStr.begin(), seedStr.end());
    std::mt19937 generator(seed);

    // fill the array with some values
    auto& ds = Dataset<DataType>::dataset;
    std::generate(ds.begin(), ds.end(), generator);
}

template <typename DataType>
InvertedRange<DataType>::InvertedRange(std::size_t size)
    : Dataset<DataType>(size)
{
    auto& ds = Dataset<DataType>::dataset;
	std::iota(ds.begin(), ds.end(), std::numeric_limits<DataType>::min());
	std::reverse(ds.begin(), ds.end());
}

template <typename DataType>
Range<DataType>::Range(std::size_t size)
    : Dataset<DataType>(size)
{
    auto& ds = Dataset<DataType>::dataset;
	std::iota(ds.begin(), ds.end(), std::numeric_limits<DataType>::min());
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
