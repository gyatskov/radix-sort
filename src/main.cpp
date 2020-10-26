#include "CRunner.h"

#include <iostream>

int main(int argc, char** argv)
{
    Arguments arguments(argc, argv);
	CRunner radixSortRunner(arguments);

    radixSortRunner.EnterMainLoop();

	std::cout << "Press any key..." << std::endl;
	std::cin.get();
}
