[![CMake on a single platform](https://github.com/gyatskov/radix-sort/actions/workflows/cmake-single-platform.yml/badge.svg)](https://github.com/gyatskov/radix-sort/actions/workflows/cmake-single-platform.yml)
# radix-sort #
GPU optimized implementation of Radix Sort using OpenCL.

# Supported compilers / platforms #
## Compilers ##
C++20-enabled compilers are supported, e.g.:

 * GCC 14
 * Clang 16
 * Visual Studio 2019

## Build Tool ##
 
 * cmake 4.0+

## Platforms ##
Every OpenCL 1.2 compliant driver should be supported. For NVIDIA devices, install CUDA drivers.

# Building #
Libraries and tests can be built as follows:

```shell
git clone github.com/gyatskov/radix-sort
cd radix-sort
mkdir build
cmake  -H. -B build
cmake --build build
```

Tests and RadixSort.cl kernel will be installed to `build/tests`.

# Unit Tests #
Run
```
ctest --test-dir build/tests --output-on-failure
```

# Documentation #
The implementation is based on papers referenced in [doc.pdf](doc/doc.pdf)
