#pragma once

#include <string>
#include <vector>

class Arguments {
private:
    std::vector<std::string> arguments;

public:
    Arguments(int argc, char* argv[]);
    
    Arguments();

    std::vector<std::string> getArguments();
};