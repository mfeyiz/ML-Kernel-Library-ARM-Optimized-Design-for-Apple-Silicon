#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <os/signpost.h>
#include "gemm.h"

static double calculateGFLOPS(size_t M, size_t K, size_t N, double elapsed_seconds) {
    return (2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K)) / (elapsed_seconds * 1e9);
}

static os_log_t bench_log() {
    static os_log_t logh = os_log_create("hwml", "bench_gemm");
    return logh;
}

#define HWML_TIMED_MS_SIGNPOST(NAME_LITERAL, EXPR)                                 \
    ([&]() -> double {                                                            \
        os_log_t _logh = bench_log();                                             \
        os_signpost_id_t _sid = os_signpost_id_generate(_logh);                   \
        os_signpost_interval_begin(_logh, _sid, NAME_LITERAL);                    \
        auto _start = std::chrono::high_resolution_clock::now();                  \
        EXPR;                                                                     \
        auto _end = std::chrono::high_resolution_clock::now();                    \
        os_signpost_interval_end(_logh, _sid, NAME_LITERAL);                      \
        return std::chrono::duration<double, std::milli>(_end - _start).count(); \
    }())

int main() {
    const std::vector<int> sizes = {512, 1024, 2048};
    constexpr int WARMUP = 1;
    constexpr int ITERS = 3;

    std::cout << "hwml bench_gemm (Apple Silicon)" << std::endl;
    std::cout << "L2 cache (reported): " << hwml::get_cache_size() / 1024 << " KB" << std::endl;
    std::cout << "hardware_concurrency: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "Note: Use Instruments -> Counters and filter signposts (hwml/bench_gemm)." << std::endl;
    std::cout << std::endl;
    
    std::cout << std::setw(6) << "N"
              << std::setw(10) << "Naive"
              << std::setw(10) << "Tiled"
              << std::setw(10) << "NEON"
              << std::setw(10) << "MT"
              << std::setw(12) << "Accel"
              << std::setw(10) << "Speedup"
              << std::setw(10) << "MT_GFL"
              << std::setw(10) << "Acc_GFL" << std::endl;
    std::cout << std::string(88, '-') << std::endl;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int N : sizes) {
        std::vector<float> A(N * N);
        std::vector<float> B(N * N);
        
        for (int i = 0; i < N * N; ++i) {
            A[i] = dist(gen);
            B[i] = dist(gen);
        }
        
        std::vector<float> C_naive(N * N, 0.0f);
        double time_naive = HWML_TIMED_MS_SIGNPOST("gemm_naive", do {
            for (int w = 0; w < WARMUP; ++w) {
                hwml::gemm_naive(A.data(), B.data(), C_naive.data(), N, N, N);
            }
            for (int it = 0; it < ITERS; ++it) {
                hwml::gemm_naive(A.data(), B.data(), C_naive.data(), N, N, N);
            }
        } while (0));
        time_naive /= ITERS;
        
        std::vector<float> C_tiled(N * N, 0.0f);
        double time_tiled = HWML_TIMED_MS_SIGNPOST("gemm_tiled", do {
            for (int w = 0; w < WARMUP; ++w) {
                hwml::gemm_tiled(A.data(), B.data(), C_tiled.data(), N, N, N);
            }
            for (int it = 0; it < ITERS; ++it) {
                hwml::gemm_tiled(A.data(), B.data(), C_tiled.data(), N, N, N);
            }
        } while (0));
        time_tiled /= ITERS;
        
        std::vector<float> C_neon(N * N, 0.0f);
        double time_neon = HWML_TIMED_MS_SIGNPOST("gemm_neon", do {
            for (int w = 0; w < WARMUP; ++w) {
                hwml::gemm_neon(A.data(), B.data(), C_neon.data(), N, N, N);
            }
            for (int it = 0; it < ITERS; ++it) {
                hwml::gemm_neon(A.data(), B.data(), C_neon.data(), N, N, N);
            }
        } while (0));
        time_neon /= ITERS;
        
        std::vector<float> C_mt(N * N, 0.0f);
        double time_mt = HWML_TIMED_MS_SIGNPOST("gemm_mt", do {
            for (int w = 0; w < WARMUP; ++w) {
                hwml::gemm_mt(A.data(), B.data(), C_mt.data(), N, N, N);
            }
            for (int it = 0; it < ITERS; ++it) {
                hwml::gemm_mt(A.data(), B.data(), C_mt.data(), N, N, N);
            }
        } while (0));
        time_mt /= ITERS;

        std::vector<float> C_accel(N * N, 0.0f);
        double time_accel = HWML_TIMED_MS_SIGNPOST("gemm_accelerate", do {
            for (int w = 0; w < WARMUP; ++w) {
                hwml::gemm_accelerate(A.data(), B.data(), C_accel.data(), N, N, N);
            }
            for (int it = 0; it < ITERS; ++it) {
                hwml::gemm_accelerate(A.data(), B.data(), C_accel.data(), N, N, N);
            }
        } while (0));
        time_accel /= ITERS;
        
        double gflops_mt = calculateGFLOPS(N, N, N, time_mt / 1000.0);
        double gflops_accel = calculateGFLOPS(N, N, N, time_accel / 1000.0);
        double speedup_mt = time_naive / time_mt;
        
        std::cout << std::setw(6) << N
                  << std::fixed << std::setprecision(2)
                  << std::setw(10) << time_naive
                  << std::setw(10) << time_tiled
                  << std::setw(10) << time_neon
                  << std::setw(10) << time_mt
                  << std::setw(12) << time_accel
                  << std::setprecision(2) << std::setw(9) << speedup_mt << "x"
                  << std::setw(10) << gflops_mt
                  << std::setw(10) << gflops_accel << std::endl;
    }
    
    return 0;
}