#include "gemm.h"
#include <algorithm>
#include <arm_neon.h>

namespace hwml {

void gemm_neon(const float* A, const float* B, float* C, int N) {
    constexpr int BLOCK_SIZE = 64;
    
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                int i_max = std::min(i + BLOCK_SIZE, N);
                int j_max = std::min(j + BLOCK_SIZE, N);
                int k_max = std::min(k + BLOCK_SIZE, N);
                
                for (int ii = i; ii < i_max; ++ii) {
                    for (int jj = j; jj < j_max; ++jj) {
                        float sum = 0.0f;
                        
                        for (int kk = k; kk < k_max; ++kk) {
                            sum += A[ii * N + kk] * B[kk * N + jj];
                        }
                        
                        C[ii * N + jj] += sum;
                    }
                }
            }
        }
    }
}

} // namespace hwml