#pragma once

#include "Parameters.h"

#include <string>
#include <vector>

struct RadixSortOptions
{
    /// Number of actual elements
    std::size_t num_elements;
    bool perf_to_stdout;
    bool perf_to_csv;
    bool perf_csv_to_stdout;
    bool verbose;

    explicit RadixSortOptions(std::vector<std::string> args) :
        num_elements(AlgorithmParameters<float>::_NUM_MAX_INPUT_ELEMS),
        perf_to_stdout(false),
        perf_to_csv(false),
        perf_csv_to_stdout(false),
        verbose(false)
    {
        for (std::size_t i = 0; i < args.size(); i++) {
            auto arg = args[i];
            if (arg == "--num-elements") {
                num_elements = std::stoi(args[i + 1]);
                i++;
            } else if (arg == "--perf-to-stdout") {
                perf_to_stdout = true;
            } else if (arg == "--perf-to-csv") {
                perf_to_csv = true;
            } else if (arg == "--perf-csv-to-stdout") {
                perf_csv_to_stdout = true;
            } else if (arg == "-v" || arg == "--verbose") {
                verbose = true;
            }
        }
    }
};
