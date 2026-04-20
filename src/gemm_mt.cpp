#include "gemm.h"
#include <algorithm>
#include <arm_neon.h>
#include <cstddef>
#include <dispatch/dispatch.h>

namespace hwml {

void gemm_neon(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha, float beta);

void gemm_mt(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha, float beta) {
    if (beta == 0.0f) {
        std::fill_n(C, M * N, 0.0f);
    } else if (beta != 1.0f) {
        for (size_t idx = 0; idx < M * N; ++idx) {
            C[idx] = beta * C[idx];
        }
    }
    
    if (alpha == 0.0f) return;
    
    constexpr size_t BLOCK_SIZE = 64;
    
    size_t num_m_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dispatch_apply(num_m_blocks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t block_i) {
        size_t i = block_i * BLOCK_SIZE;
        size_t i_max = std::min(i + BLOCK_SIZE, M);
        
        for (size_t k = 0; k < K; k += BLOCK_SIZE) {
            size_t k_max = std::min(k + BLOCK_SIZE, K);
            for (size_t j = 0; j < N; j += BLOCK_SIZE) {
                size_t j_max = std::min(j + BLOCK_SIZE, N);
                
                size_t ii = i;
                for (; ii + 3 < i_max; ii += 4) {
                    size_t jj = j;
                    for (; jj + 3 < j_max; jj += 4) {
                        float32x4_t c0 = vld1q_f32(&C[ii * N + jj]);
                        float32x4_t c1 = vld1q_f32(&C[(ii + 1) * N + jj]);
                        float32x4_t c2 = vld1q_f32(&C[(ii + 2) * N + jj]);
                        float32x4_t c3 = vld1q_f32(&C[(ii + 3) * N + jj]);
                        
                        for (size_t kk = k; kk < k_max; ++kk) {
                            float32x4_t b_vec = vld1q_f32(&B[kk * N + jj]);
                            
                            c0 = vmlaq_f32(c0, vdupq_n_f32(alpha * A[ii * K + kk]), b_vec);
                            c1 = vmlaq_f32(c1, vdupq_n_f32(alpha * A[(ii + 1) * K + kk]), b_vec);
                            c2 = vmlaq_f32(c2, vdupq_n_f32(alpha * A[(ii + 2) * K + kk]), b_vec);
                            c3 = vmlaq_f32(c3, vdupq_n_f32(alpha * A[(ii + 3) * K + kk]), b_vec);
                        }
                        
                        vst1q_f32(&C[ii * N + jj], c0);
                        vst1q_f32(&C[(ii + 1) * N + jj], c1);
                        vst1q_f32(&C[(ii + 2) * N + jj], c2);
                        vst1q_f32(&C[(ii + 3) * N + jj], c3);
                    }
                    
                    for (; jj < j_max; ++jj) {
                        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                        for (size_t kk = k; kk < k_max; ++kk) {
                            float b_val = B[kk * N + jj];
                            sum0 += A[ii * K + kk] * b_val;
                            sum1 += A[(ii + 1) * K + kk] * b_val;
                            sum2 += A[(ii + 2) * K + kk] * b_val;
                            sum3 += A[(ii + 3) * K + kk] * b_val;
                        }
                        C[ii * N + jj] += alpha * sum0;
                        C[(ii + 1) * N + jj] += alpha * sum1;
                        C[(ii + 2) * N + jj] += alpha * sum2;
                        C[(ii + 3) * N + jj] += alpha * sum3;
                    }
                }
                
                for (; ii < i_max; ++ii) {
                    for (size_t kk = k; kk < k_max; ++kk) {
                        float a_val = alpha * A[ii * K + kk];
                        float32x4_t a_vec = vdupq_n_f32(a_val);
                        
                        size_t jj = j;
                        for (; jj + 3 < j_max; jj += 4) {
                            float32x4_t b_vec = vld1q_f32(&B[kk * N + jj]);
                            float32x4_t c_vec = vld1q_f32(&C[ii * N + jj]);
                            c_vec = vmlaq_f32(c_vec, a_vec, b_vec);
                            vst1q_f32(&C[ii * N + jj], c_vec);
                        }
                        
                        for (; jj < j_max; ++jj) {
                            C[ii * N + jj] += a_val * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    });
}

void gemm(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha, float beta) {
    if (M >= 512 || N >= 512 || K >= 512) {
        gemm_mt(A, B, C, M, K, N, alpha, beta);
    } else {
        gemm_neon(A, B, C, M, K, N, alpha, beta);
    }
}

} // namespace hwml
