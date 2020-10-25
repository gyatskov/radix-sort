# radix-sort #
GPU optimized implementation of Radix Sort using OpenCL.

# Supported compilers / platforms #
## Compilers ##
GCC 9 has been tested on Ubuntu 20.04.

## Platforms ##
Every OpenCL 1.2 compliant driver should be supported. For NVIDIA devices, CUDA drivers may be necessary.

# Building #
Building is performed using CMake.

```
git clone github.com/gyatskov/radix-sort
cd radix-sort
cmake -DCMAKE_INSTALL_PREFIX=. -B build -S radix-sort
cmake --build build
cmake --build build --target install
```

# Running tests #
```
cd bin
./radixsort
```

# Documentation #
The implementation is based on papers referenced in [doc.pdf](doc/doc.pdf)
