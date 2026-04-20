#include "gemm.h"
#include <arm_neon.h>
#include <cstddef>

namespace hwml {

void gemm_naive(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha, float beta) {
    for (size_t i = 0; i < M * N; ++i) {
        C[i] = beta * C[i];
    }
    
    if (alpha == 0.0f) return;
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a = alpha * A[i * K + k];
            float32x4_t a_vec = vdupq_n_f32(a);
            
            size_t j = 0;
            for (; j + 3 < N; j += 4) {
                float32x4_t b_vec = vld1q_f32(&B[k * N + j]);
                float32x4_t c_vec = vld1q_f32(&C[i * N + j]);
                c_vec = vmlaq_f32(c_vec, a_vec, b_vec);
                vst1q_f32(&C[i * N + j], c_vec);
            }
            
            for (; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

} // namespace hwml