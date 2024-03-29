[![CMake on a single platform](https://github.com/gyatskov/radix-sort/actions/workflows/cmake-single-platform.yml/badge.svg)](https://github.com/gyatskov/radix-sort/actions/workflows/cmake-single-platform.yml)
# radix-sort #
GPU optimized implementation of Radix Sort using OpenCL.

# Supported compilers / platforms #
## Compilers ##
C++17-enabled compilers are supported, e.g.:

 * GCC 9.3
 * Clang 10
 * Visual Studio 2019

## Build Tool ##
 
 * cmake 3.17+

## Platforms ##
Every OpenCL 1.2 compliant driver should be supported. For NVIDIA devices, CUDA drivers may be necessary.

# Building #
Libraries and tests can be built as follows:

Linux:
```bash
git clone github.com/gyatskov/radix-sort
cd radix-sort
mkdir build
cmake -DCMAKE_INSTALL_PREFIX="bin/" -H. -B build
cmake --build build
cmake --build build --target install
```

Windows (PowerShell):
```powershell
git clone github.com/gyatskov/radix-sort
cd radix-sort
mkdir build
cmake -DCMAKE_INSTALL_PREFIX="$(Get-Location)/bin" -H. -B build
cmake --build build
cmake --build build --target install
```

Tests will be installed to `build/tests`.

Libraries (static libradixsortcl as well as dynamic libOpenCL) will be installed to `bin`.

# Unit Tests #
Run
```
ctest --test-dir build/tests --output-on-failure
```

# Documentation #
The implementation is based on papers referenced in [doc.pdf](doc/doc.pdf)
