# AGENTS.md

Developer notes and a lightweight roadmap for `arm_gemm_apple`.

## Current state

Implemented and wired into build/tests/benchmarks:

- Multiple FP32 GEMM paths: naive, tiled+packing, NEON-blocked, GCD MT
- Apple Accelerate baseline: `gemm_accelerate` (CBLAS `cblas_sgemm`)
- In-place activations: ReLU, sigmoid (NEON + GCD MT)
- C++ unit-style tests and C++ benchmarks
- Python bindings with input validation + demo scripts

Benchmark tooling:

- `benchmarks/bench_gemm.cpp` emits `os_signpost` intervals for Instruments
  counter measurements per-kernel.
- `tests/benchmark_only.py` produces `benchmark_results.txt` and appends
  `tests/ml_demo.py` output for demo-style reporting.

## Roadmap (suggested next work)

1) Rectangular GEMM benchmarking in C++ (`bench_gemm` currently uses N×N cases)
2) Improve Python alpha/beta semantics by allowing an input/output C buffer
   (today Python wrappers return fresh C, so beta is effectively ignored)
3) GEMM packing and microkernel cleanup
   - consider packing for `gemm_neon` to reduce B streaming pressure
   - investigate prefetching and better block sizes per-cache
4) Add more profiler-backed metrics in the write-up
   - Instruments Counters scoped to signpost intervals (L1/L2 misses, cycles)

## Notes

- Treat Accelerate/NumPy results as an upper-bound reference on Apple Silicon.
  Vendor libraries may use implementation paths not comparable to public NEON.
