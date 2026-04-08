#ifndef GEMM_H
#define GEMM_H

// Expects 1D flat C-contiguous arrays (NumPy compatible).
// Memory layout: row-major (row * N + col)

namespace hwml {

void gemm_naive(const float* A, const float* B, float* C, int N);
void gemm_tiled(const float* A, const float* B, float* C, int N);
void gemm_neon(const float* A, const float* B, float* C, int N);
void gemm_mt(const float* A, const float* B, float* C, int N);

} // namespace hwml

#endif // GEMM_H