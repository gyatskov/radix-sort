#pragma once

#include <cstdint>

/// Non-owning reference to memory segment
template <typename DataType>
struct CheapSpan
{
    using pointer = DataType*;
    using size_type = std::size_t;

    // Start location of memory
    pointer data{nullptr};

    // length in elements of DataType
    size_type length{0U};
};
