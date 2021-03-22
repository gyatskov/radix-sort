#pragma once

#include <cstdint>
#include <limits>

struct Statistics
{
    double min{std::numeric_limits<decltype(min)>::infinity()};
    double max{-std::numeric_limits<decltype(max)>::infinity()};
    double avg{0.0};
    double sum{0.0};

    /** Sample count **/
    std::size_t n{0U};

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
