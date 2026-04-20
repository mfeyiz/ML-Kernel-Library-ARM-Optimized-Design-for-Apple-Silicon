#include <iostream>
#include <vector>
#include <cmath>
#include "gemm.h"

bool floatEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}

int main() {
    const int N = 3;
    
    std::vector<float> A = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    
    std::vector<float> B = {
        1.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 3.0f
    };
    
    std::vector<float> C(N * N, 0.0f);
    
    std::vector<float> expected = {
        1.0f,  4.0f,  9.0f,
        4.0f, 10.0f, 18.0f,
        7.0f, 16.0f, 27.0f
    };
    
    hwml::gemm_naive(A.data(), B.data(), C.data(), N, N, N);
    
    bool passed = true;
    for (int i = 0; i < N * N; ++i) {
        if (!floatEqual(C[i], expected[i])) {
            std::cout << "FAIL: C[" << i << "] = " << C[i] 
                      << ", expected " << expected[i] << std::endl;
            passed = false;
        }
    }
    
    if (passed) {
        std::cout << "PASS: gemm_naive correctness validation" << std::endl;
    }
    
    std::vector<float> C2(N * N, 0.0f);
    hwml::gemm_tiled(A.data(), B.data(), C2.data(), N, N, N);
    
    bool passed2 = true;
    for (int i = 0; i < N * N; ++i) {
        if (!floatEqual(C2[i], expected[i])) {
            std::cout << "FAIL: gemm_tiled C[" << i << "] = " << C2[i] 
                      << ", expected " << expected[i] << std::endl;
            passed2 = false;
        }
    }
    
    if (passed2) {
        std::cout << "PASS: gemm_tiled correctness validation" << std::endl;
    }
    
    std::vector<float> C3(N * N, 0.0f);
    hwml::gemm_neon(A.data(), B.data(), C3.data(), N, N, N);
    
    bool passed3 = true;
    for (int i = 0; i < N * N; ++i) {
        if (!floatEqual(C3[i], expected[i])) {
            std::cout << "FAIL: gemm_neon C[" << i << "] = " << C3[i] 
                      << ", expected " << expected[i] << std::endl;
            passed3 = false;
        }
    }
    
    if (passed3) {
        std::cout << "PASS: gemm_neon correctness validation" << std::endl;
    }
    
    std::vector<float> C4(N * N, 0.0f);
    hwml::gemm_mt(A.data(), B.data(), C4.data(), N, N, N);
    
    bool passed4 = true;
    for (int i = 0; i < N * N; ++i) {
        if (!floatEqual(C4[i], expected[i])) {
            std::cout << "FAIL: gemm_mt C[" << i << "] = " << C4[i] 
                      << ", expected " << expected[i] << std::endl;
            passed4 = false;
        }
    }
    
    if (passed4) {
        std::cout << "PASS: gemm_mt correctness validation" << std::endl;
    }
    
    return (passed && passed2 && passed3 && passed4) ? 0 : 1;
}