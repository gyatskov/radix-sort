#pragma once

#include <cstdint>
#include <limits>

struct Statistics
{
    double min;
    double max;
    double avg;
    double sum;

    std::size_t n;

    Statistics() :
        min(std::numeric_limits<decltype(min)>::infinity()),
        max(-std::numeric_limits<decltype(max)>::infinity()),
        avg(0),
        sum(0),
        n(0)
    {}

    void update(double value) {
        n++;
        sum += value;
        avg = sum / n;
        if (value > max) {
            max = value;
        }
        else if (value < min) {
            min = value;
        }
    }
};
