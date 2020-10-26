#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "CRunner.h"

TEST_CASE( "Main test", "[main]" )
{
    // Non-interactive mode
    constexpr auto argc = 0;
    char* argv[] = {};

    Arguments arguments(argc, argv);
	CRunner radixSortRunner(arguments);

    REQUIRE(radixSortRunner.EnterMainLoop());
}
