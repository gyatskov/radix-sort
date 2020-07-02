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
cd radix-sort
mkdir ../build
cmake -B ../build
cmake --build ../build
```

# Running tests #
```
cd radix-sort
../build/Assignment
```

# Implementation #
The implementation is based on papers referenced in [doc.pdf](doc/doc.pdf)
