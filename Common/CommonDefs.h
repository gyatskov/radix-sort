#pragma once

#define _STR(x)      #x

// Custom pragma for msvc
#if defined(_MSC_VER)
    #define TODO(x)      __pragma(message ("TODO - " _STR(x) " :: " __FILE__ " @ " _STR( __LINE__ )))
#else
    #define DO_PRAGMA(x) _Pragma ( _STR(x) )
    #define TODO(x)      DO_PRAGMA(message( "TODO - " _STR(x) " :: " __FILE__ " @ " _STR( __LINE__ )))
#endif

// OpenCL specifics
// Use OpenCL 1.2
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
