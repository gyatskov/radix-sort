#pragma once

#include <string>
#include <vector>

class Arguments {
public:
    Arguments(int argc, char* argv[]);

    Arguments() = default;
    ~Arguments() = default;
    Arguments(Arguments&&) = default;
    Arguments(const Arguments&) = default;
    Arguments& operator=(Arguments&) = default;
    Arguments& operator=(Arguments&&) = default;

    std::vector<std::string> getArguments();
private:
    std::vector<std::string> arguments;

};
