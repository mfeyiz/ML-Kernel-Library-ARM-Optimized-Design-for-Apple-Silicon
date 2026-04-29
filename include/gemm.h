#ifndef GEMM_H
#define GEMM_H

#include <cstddef>

namespace hwml {

size_t get_cache_size();

void gemm_naive(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha = 1.0f, float beta = 0.0f);
void gemm_tiled(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha = 1.0f, float beta = 0.0f);
void gemm_neon(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha = 1.0f, float beta = 0.0f);
void gemm_mt(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha = 1.0f, float beta = 0.0f);

// Accelerate / CBLAS reference (upper-bound baseline on Apple Silicon)
void gemm_accelerate(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha = 1.0f, float beta = 0.0f);

void gemm(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha = 1.0f, float beta = 0.0f);

} // namespace hwml

#endif // GEMM_H