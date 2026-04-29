# arm_gemm_apple

Hardware-aware ML kernel library for Apple Silicon (CPU-side).

This repository implements several FP32 GEMM paths and a small set of activation
kernels. It is designed for demo and research-style evaluation on Apple Silicon
macOS systems where the publicly programmable SIMD target is ARM NEON.

An explicit Apple Accelerate (BLAS) path is included as a practical upper-bound
reference. It is not treated as a like-for-like “NEON-only” comparison.

## What’s inside

### GEMM (FP32)

All C++ GEMM entry points support general rectangular shapes:
`A (M×K) @ B (K×N) -> C (M×N)`.

- `hwml::gemm_naive`: simple baseline loop nest with NEON vectorization in the
	inner `j` dimension.
- `hwml::gemm_tiled`: cache-blocked GEMM with on-the-fly packing of A and B
	panels into contiguous buffers.
- `hwml::gemm_neon`: blocked GEMM with explicit NEON `vld1q_f32`/`vmlaq_f32`.
- `hwml::gemm_mt`: multi-threaded GEMM using Grand Central Dispatch
	(`dispatch_apply`) over M-tiles.
- `hwml::gemm_accelerate`: Apple Accelerate / CBLAS `cblas_sgemm` reference.
- `hwml::gemm`: simple auto-select (prefers MT for larger problems).

Alpha/beta are supported in the C++ API:

`C = alpha * A @ B + beta * C`

### Activations (FP32, in-place)

- ReLU: `hwml::relu_neon`, `hwml::relu_mt`
- Sigmoid (approx exp): `hwml::sigmoid_neon`, `hwml::sigmoid_mt`

## Build requirements

- Apple Silicon host (arm64)
- macOS 12+
- Xcode Command Line Tools / Apple clang
- CMake 3.15+
- Python + pybind11 (CMake finds your installed pybind11)

OpenMP is optional; this repo uses GCD for threading.

## Build (CMake)

Out-of-source build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(sysctl -n hw.ncpu)"
```

Build outputs (examples):

- `build/libhwml_kernels.a`
- `build/arm_gemm_apple.*.so` (Python extension module)
- `build/test_gemm`, `build/test_activations`
- `build/bench_gemm`, `build/bench_activations`

## Run correctness tests

### C++ tests

```bash
./build/test_gemm
./build/test_activations
```

### Python binding test suite

```bash
PYTHONPATH=build python3 tests/test_python_bridge.py
```

## Demo: run benchmarks and produce a single report file

The easiest demo path is the Python benchmark script, which writes a single
`benchmark_results.txt` file containing:

1) GEMM timing table (includes `gemm_accelerate` and NumPy)
2) ML demo output (`tests/ml_demo.py`) appended under the table

Run:

```bash
PYTHONPATH=build python3 tests/benchmark_only.py
cat benchmark_results.txt
```

## C++ benchmark (with Instruments-friendly signposts)

The C++ GEMM benchmark prints a table and emits `os_signpost` intervals around
each kernel call so you can isolate counters per-kernel in Instruments.

Run:

```bash
./build/bench_gemm
```

### Measuring cache misses / counters (recommended)

On macOS, the cleanest way to report cache behavior is Instruments Counters,
scoped to the signpost intervals:

1) Open Instruments
2) Choose the “Counters” template (or a CPU counters template available)
3) Launch `build/bench_gemm`
4) Filter by subsystem/category `hwml / bench_gemm`
5) Select the signpost interval for the kernel you want (e.g. `gemm_mt`)
6) Read L1/L2 miss metrics, cycles, instructions for that time range

This approach avoids unreliable ad-hoc “cache hit/miss” estimates inside code.

## Notes on interpreting Accelerate results

- `numpy` on Apple Silicon typically uses Accelerate-backed BLAS.
- `hwml::gemm_accelerate` calls Accelerate BLAS directly.
- Treat these as a practical ceiling reference, not a NEON-only baseline.

## Python API notes (important for demos)

- GEMM functions return a new output array `C`.
- The Python wrappers currently allocate `C` as zeros before calling the C++
	kernels. That means `beta` has no effect in Python today (because it scales a
	zero-initialized `C`).
- In C++ (pointer API), `beta` behaves as expected because you control the input
	contents of `C`.

## File layout

```text
include/
	gemm.h
	activations.h
src/
	gemm_naive.cpp
	gemm_tiled.cpp
	gemm_neon.cpp
	gemm_mt.cpp
	gemm_accelerate.cpp
	activations.cpp
	bindings.cpp
	cache_utils.cpp
benchmarks/
	bench_gemm.cpp
	bench_activations.cpp
tests/
	test_gemm.cpp
	test_activations.cpp
	test_python_bridge.py
	benchmark_only.py
	ml_demo.py
```
