#pragma once

#include <cstdint>
#include <limits>

struct Statistics
{
    // Returns minimum value
    double min{std::numeric_limits<decltype(min)>::infinity()};
    // Returns maximum value
    double max{-std::numeric_limits<decltype(max)>::infinity()};
    // Returns average value
    double avg{0.0};
    // Returns sum of values
    double sum{0.0};

    /** Sample count **/
    std::size_t n{0U};

    /** Adds value to statistic **/
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
