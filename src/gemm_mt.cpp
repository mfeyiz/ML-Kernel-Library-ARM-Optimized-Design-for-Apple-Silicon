#include "gemm.h"
#include <algorithm>
#include <arm_neon.h>
#include <thread>
#include <vector>

namespace hwml {

static void gemm_mt_worker(const float* A, const float* B, float* C, int N, 
                            int row_start, int row_end) {
    constexpr int BLOCK_SIZE = 64;
    
    for (int i = row_start; i < row_end; i += BLOCK_SIZE) {
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

void gemm_mt(const float* A, const float* B, float* C, int N) {
    unsigned int max_threads = std::thread::hardware_concurrency();
    if (max_threads == 0) max_threads = 1;
    
    unsigned int num_threads = std::min(max_threads, 8u);
    
    if (N < 512 || num_threads == 1) {
        gemm_mt_worker(A, B, C, N, 0, N);
        return;
    }
    
    std::vector<std::thread> threads;
    
    int rows_per_thread = N / num_threads;
    int remainder = N % num_threads;
    
    int current_row = 0;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int row_start = current_row;
        int row_end = row_start + rows_per_thread + (t < remainder ? 1 : 0);
        current_row = row_end;
        
        threads.push_back(std::thread(gemm_mt_worker, A, B, C, N, row_start, row_end));
    }
    
    for (auto& th : threads) {
        th.join();
    }
}

} // namespace hwml