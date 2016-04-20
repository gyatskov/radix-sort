#include "CRunner.h"

#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    Arguments arguments(argc, argv);
	CRunner radixSortRunner(arguments);

    radixSortRunner.EnterMainLoop();

	cout<<"Press any key..."<<endl;
	cin.get();
}
