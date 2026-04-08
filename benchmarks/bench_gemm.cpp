#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include "gemm.h"

double calculateGFLOPS(int N, double elapsed_seconds) {
    return (2.0 * N * N * N) / (elapsed_seconds * 1e9);
}

int main() {
    const std::vector<int> sizes = {512, 1024, 2048};
    
    std::cout << std::setw(6) << "N"
              << std::setw(10) << "Naive"
              << std::setw(10) << "Tiled"
              << std::setw(10) << "NEON"
              << std::setw(10) << "MT"
              << std::setw(10) << "Speedup"
              << std::setw(10) << "MT_GFLOPS" << std::endl;
    std::cout << std::string(66, '-') << std::endl;
    
    std::random_device rd;
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
        auto start_naive = std::chrono::high_resolution_clock::now();
        hwml::gemm_naive(A.data(), B.data(), C_naive.data(), N);
        auto end_naive = std::chrono::high_resolution_clock::now();
        double time_naive = std::chrono::duration<double, std::milli>(end_naive - start_naive).count();
        
        std::vector<float> C_tiled(N * N, 0.0f);
        auto start_tiled = std::chrono::high_resolution_clock::now();
        hwml::gemm_tiled(A.data(), B.data(), C_tiled.data(), N);
        auto end_tiled = std::chrono::high_resolution_clock::now();
        double time_tiled = std::chrono::duration<double, std::milli>(end_tiled - start_tiled).count();
        
        std::vector<float> C_neon(N * N, 0.0f);
        auto start_neon = std::chrono::high_resolution_clock::now();
        hwml::gemm_neon(A.data(), B.data(), C_neon.data(), N);
        auto end_neon = std::chrono::high_resolution_clock::now();
        double time_neon = std::chrono::duration<double, std::milli>(end_neon - start_neon).count();
        
        std::vector<float> C_mt(N * N, 0.0f);
        auto start_mt = std::chrono::high_resolution_clock::now();
        hwml::gemm_mt(A.data(), B.data(), C_mt.data(), N);
        auto end_mt = std::chrono::high_resolution_clock::now();
        double time_mt = std::chrono::duration<double, std::milli>(end_mt - start_mt).count();
        
        double gflops_naive = calculateGFLOPS(N, time_naive / 1000.0);
        double gflops_tiled = calculateGFLOPS(N, time_tiled / 1000.0);
        double gflops_neon = calculateGFLOPS(N, time_neon / 1000.0);
        double gflops_mt = calculateGFLOPS(N, time_mt / 1000.0);
        double speedup_mt = time_naive / time_mt;
        
        std::cout << std::setw(6) << N
                  << std::fixed << std::setprecision(2)
                  << std::setw(10) << time_naive
                  << std::setw(10) << time_tiled
                  << std::setw(10) << time_neon
                  << std::setw(10) << time_mt
                  << std::setprecision(2) << std::setw(9) << speedup_mt << "x"
                  << std::setw(10) << gflops_mt << std::endl;
    }
    
    return 0;
}