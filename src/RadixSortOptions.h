#pragma once

#include "Parameters.h"

#include <string>
#include <vector>

struct RadixSortOptions
{
    /// Number of keys.
    std::size_t num_elements = AlgorithmConfiguration::_NUM_MAX_INPUT_ELEMS;
    /// Number of iterations for performance testing
    std::size_t num_iterations = 5;
    bool perf_to_stdout = false;
    bool perf_to_csv = false;
    bool perf_csv_to_stdout = false;
    bool verbose = false;
};

inline RadixSortOptions ParseArgs(const std::vector<std::string>& args)
{
    RadixSortOptions  result;
    for (std::size_t i = 0; i < args.size(); i++) {
        auto arg = args[i];
        if (arg == "--num-elements") {
            result.num_elements = std::stoi(args[i + 1]);
            i++;
        } else if (arg == "--perf-to-stdout") {
            result.perf_to_stdout = true;
        } else if (arg == "--perf-to-csv") {
            result.perf_to_csv = true;
        } else if (arg == "--perf-csv-to-stdout") {
            result.perf_csv_to_stdout = true;
        } else if (arg == "-v" || arg == "--verbose") {
            result.verbose = true;
        }
    }
    return result;
}
