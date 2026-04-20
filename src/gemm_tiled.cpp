#include "gemm.h"
#include <algorithm>
#include <arm_neon.h>
#include <cstddef>
#include <vector>

namespace hwml {

static void pack_A(const float* A, float* packA, size_t K, size_t i_start, size_t i_end, size_t k_start, size_t k_end) {
    size_t idx = 0;
    for (size_t i = i_start; i < i_end; ++i) {
        for (size_t k = k_start; k < k_end; ++k) {
            packA[idx++] = A[i * K + k];
        }
    }
}

static void pack_B(const float* B, float* packB, size_t N, size_t k_start, size_t k_end, size_t j_start, size_t j_end) {
    size_t idx = 0;
    for (size_t k = k_start; k < k_end; ++k) {
        for (size_t j = j_start; j < j_end; ++j) {
            packB[idx++] = B[k * N + j];
        }
    }
}

void gemm_tiled(const float* A, const float* B, float* C, size_t M, size_t K, size_t N, float alpha, float beta) {
    constexpr size_t BLOCK_MC = 64;
    constexpr size_t BLOCK_KC = 64;
    constexpr size_t BLOCK_NC = 64;
    
    if (beta == 0.0f) {
        std::fill_n(C, M * N, 0.0f);
    } else if (beta != 1.0f) {
        for (size_t i = 0; i < M * N; ++i) {
            C[i] = beta * C[i];
        }
    }
    
    if (alpha == 0.0f) return;
    
    std::vector<float> packA(BLOCK_MC * BLOCK_KC);
    std::vector<float> packB(BLOCK_KC * BLOCK_NC);
    
    for (size_t k = 0; k < K; k += BLOCK_KC) {
        size_t k_max = std::min(k + BLOCK_KC, K);
        size_t kb = k_max - k;
        
        for (size_t j = 0; j < N; j += BLOCK_NC) {
            size_t j_max = std::min(j + BLOCK_NC, N);
            size_t nb = j_max - j;
            
            pack_B(B, packB.data(), N, k, k_max, j, j_max);
            
            for (size_t i = 0; i < M; i += BLOCK_MC) {
                size_t i_max = std::min(i + BLOCK_MC, M);
                size_t mb = i_max - i;
                
                pack_A(A, packA.data(), K, i, i_max, k, k_max);
                
                for (size_t ii = 0; ii < mb; ++ii) {
                    for (size_t kk = 0; kk < kb; ++kk) {
                        float a_val = alpha * packA[ii * kb + kk];
                        float32x4_t a_vec = vdupq_n_f32(a_val);
                        
                        size_t jj = 0;
                        for (; jj + 3 < nb; jj += 4) {
                            float32x4_t b_vec = vld1q_f32(&packB[kk * nb + jj]);
                            
                            size_t c_idx = (i + ii) * N + (j + jj);
                            float32x4_t c_vec = vld1q_f32(&C[c_idx]);
                            
                            c_vec = vmlaq_f32(c_vec, a_vec, b_vec);
                            vst1q_f32(&C[c_idx], c_vec);
                        }
                        
                        for (; jj < nb; ++jj) {
                            C[(i + ii) * N + (j + jj)] += a_val * packB[kk * nb + jj];
                        }
                    }
                }
            }
        }
    }
}

} // namespace hwml
