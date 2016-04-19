#pragma once

#include <vector>
#include <map>
#include <string>
#include <functional>
#include <cstdint>
#include <random>
#include <algorithm>
#include <numeric>

#include "Parameters.h"

template <typename _DataType>
struct Dataset {
	using DataType = _DataType;
	using Parameters = Parameters < DataType > ;

	static const char* const name;
	std::vector<DataType> dataset;

    Dataset(size_t size = Parameters::_NUM_MAX_INPUT_ELEMS) : dataset(size)
	{}

	virtual const char* const getName() {
		return name;
	}
};

template <typename DataType>
struct Zeros : Dataset<DataType> {
	static const char* const name;

    Zeros(size_t size = Parameters::_NUM_MAX_INPUT_ELEMS);

	virtual const char* const getName() {
		return Zeros::name;
	}
};

template <typename DataType>
struct RandomDistributed : Dataset < DataType > {
	static const char* const name;

    RandomDistributed(size_t size = Parameters::_NUM_MAX_INPUT_ELEMS);

	virtual const char* const getName() {
		return RandomDistributed::name;
	}
};

template <typename DataType>
struct Random : Dataset < DataType > {
    static const char* const name;

    Random(size_t size = Parameters::_NUM_MAX_INPUT_ELEMS);

    virtual const char* const getName() {
        return Random::name;
    }
};

template <typename DataType>
struct Range : Dataset < DataType > {
	static const char* const name;

    Range(size_t size = Parameters::_NUM_MAX_INPUT_ELEMS);

	virtual const char* const getName() {
		return Range::name;
	}
};

template <typename DataType>
struct InvertedRange : Dataset < DataType > {
	static const char* const name;

    InvertedRange(size_t size = Parameters::_NUM_MAX_INPUT_ELEMS);

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
