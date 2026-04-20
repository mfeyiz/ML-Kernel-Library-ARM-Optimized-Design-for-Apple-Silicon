# ML Kernel Library - ARM Optimized Design for Apple Silicon

## Overview
This project provides a hardware-aware Machine Learning kernel library optimized specifically for Apple Silicon (M-series) architecture. It focuses on highly efficient General Matrix Multiplication (GEMM) operations and common activation functions utilizing ARM NEON SIMD instructions, optimal cache management with data packing, and Apple's Grand Central Dispatch (GCD) for concurrency. 

The library exposes a C++ API alongside Python bindings (via Pybind11), allowing seamless integration into higher-level machine learning frameworks.

## Features
- **ARM NEON Microkernels**: Utilizes `arm_neon.h` intrinsics in a 4x4 register-blocked architecture with loop unrolling for high-throughput FMA (Fused Multiply-Add) operations.
- **Grand Central Dispatch (GCD)**: Employs `dispatch_apply` for asymmetric core-aware thread distribution, replacing standard `std::thread` overheads.
- **Arbitrary Rectangular Matrices**: All routines natively support fully unconstrained M, K, N matrix dimensions for inputs $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, and output $C \in \mathbb{R}^{M \times N}$.
- **Advanced Cache Blocking & Data Packing**: Memory access layouts are optimized by packing blocks of A and B dynamically into contiguous data segments, significantly reducing TLB misses and L1/L2 cache evictions during computation.
- **Production-Ready Python Bindings**: Zero-copy data handling and robust validation schemas guarantee safety against memory leaks and invalid rank/dimension arguments on the Python side.

## System Requirements
- Apple Silicon Host (M1/M2/M3 CPU preferred)
- macOS 12.0+ (Requires `<dispatch/dispatch.h>`)
- Clang/LLVM with standard C++17 support
- CMake >= 3.15
- Pybind11 (for Python integration)

## Build Instructions
An out-of-source CMake build is recommended:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
```

The output will include:
- `libhwml_kernels.a`: Static library containing the core logic.
- `arm_gemm_apple.*.so`: Compiled Python bindings module.
- `test_gemm`, `test_activations`, `bench_gemm`, `bench_activations`: Executable test binaries and benchmarks.

## Running Tests
C++ Unit Tests and Benchmarks:
```bash
./build/test_gemm
./build/test_activations
./build/bench_gemm
./build/bench_activations
```

Python Binding Tests:
```bash
PYTHONPATH=build python3 tests/test_python_bridge.py
```

## Python API Usage

```python
import numpy as np
import arm_gemm_apple

# Matrix Multiplication
M, K, N = 1200, 800, 1600
A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

# Dispatches an optimized execution depending on matrix dimensions
C = arm_gemm_apple.gemm_mt(A, B, alpha=1.0, beta=0.0)

# Activation Functions (In-Place Modifications)
X = np.random.randn(1000, 1000).astype(np.float32)
arm_gemm_apple.relu_mt(X)
arm_gemm_apple.sigmoid_mt(X)
```

## Internal Architecture
1. **gemm_naive**: Reference implementation.
2. **gemm_tiled**: Implements 64x64 parameter blocking with on-the-fly data re-packing (paneling) into contiguous memory regions.
3. **gemm_neon**: Integrates a 4x4 inner loop SIMD processing model.
4. **gemm_mt**: Leverages `dispatch_get_global_queue` to schedule dynamically partitioned block workloads onto available physical processing queues. 

## Contribution and Extension
When extending the code, adhere to modern C++17 memory conventions and Pybind11 reference counting rules. Ensure any algorithm tuning utilizes `size_t` for indexing to avoid 32-bit overflow on excessive tensor allocations.
