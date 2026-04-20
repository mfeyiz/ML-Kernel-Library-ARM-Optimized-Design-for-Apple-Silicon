#include "gemm.h"
#include <arm_neon.h>
#include <cstddef>

namespace hwml {

void gemm_neon(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha, float beta);

void gemm_bf16(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha, float beta) {
    gemm_neon(A, B, C, M, K, N, alpha, beta);
}

} // namespace hwml