#pragma once

#include <cstdint>
#include <string>

#include "../Common/CArguments.h"

struct RadixSortOptions {
    size_t num_elements;
    bool perf_to_stdout;
    bool perf_to_csv;
    bool perf_csv_to_stdout;
    bool verbose;

    RadixSortOptions(Arguments args) : 
        num_elements(Parameters<int>::_NUM_MAX_INPUT_ELEMS),
        perf_to_stdout(false),
        perf_to_csv(false),
        perf_csv_to_stdout(false),
        verbose(false)
    {
        for (size_t i = 0; i < args.getArguments().size(); i++) {
            auto arg = args.getArguments()[i];
            if (arg == "--num-elements") {
                num_elements = std::stoi(args.getArguments()[i + 1]);
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