# radix-sort #
GPU optimized implementation of Radix Sort using OpenCL.

# Supported compilers / platforms #
## Compilers ##
C++17-enabled compilers are supported, e.g.:

 * GCC 9.3
 * Clang 10
 * Visual Studio 2019

## Platforms ##
Every OpenCL 1.2 compliant driver should be supported. For NVIDIA devices, CUDA drivers may be necessary.

# Building #
Libraries and tests are built in a few steps:

Linux:
```bash
git clone github.com/gyatskov/radix-sort
cd radix-sort
cmake -DCMAKE_INSTALL_PREFIX=. -B build
cmake --build build
cmake --build build --target install
```

Windows (PowerShell):
```powershell
git clone github.com/gyatskov/radix-sort
cd radix-sort
cmake -DCMAKE_INSTALL_PREFIX="$(Get-Location)" -B build
cmake --build build
cmake --build build --target install
```

# Running tests #
```
(cd build/tests && ctest)
```

# Documentation #
The implementation is based on papers referenced in [doc.pdf](doc/doc.pdf)
