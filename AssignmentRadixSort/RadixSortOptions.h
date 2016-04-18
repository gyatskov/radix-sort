#pragma once

#include <cstdint>
#include <string>

#include "../Common/CArguments.h"

struct RadixSortOptions {
    size_t num_elements;

    RadixSortOptions(Arguments args) : num_elements(Parameters<int>::_NUM_MAX_INPUT_ELEMS) {
        for (size_t i = 0; i < args.getArguments().size(); i++) {
            auto arg = args.getArguments()[i];
            if (arg == "--num-elements") {
                num_elements = std::stoi(args.getArguments()[i + 1]);
                i++;
            }
        }
    }

};