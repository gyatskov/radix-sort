#include "CArguments.h"

Arguments::Arguments(int argc, char* argv[])
{
    for (int i = 0; i < argc; i++) {
        arguments.push_back(argv[i]);
    }
}

Arguments::Arguments()
{}

std::vector<std::string> Arguments::getArguments()
{
    return arguments;
}
