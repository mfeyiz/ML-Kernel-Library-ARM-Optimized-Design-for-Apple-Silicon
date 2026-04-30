#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "gemm.h"

struct Shape {
    int M;
    int K;
    int N;
};

static bool compare_outputs(const std::vector<float>& ref, const std::vector<float>& out, float rtol, float atol) {
    if (ref.size() != out.size()) {
        return false;
    }

    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::fabs(ref[i] - out[i]);
        float tol = atol + rtol * std::fabs(ref[i]);
        if (diff > tol) {
            std::cout << "FAIL: index " << i
                      << " ref=" << ref[i]
                      << " out=" << out[i]
                      << " diff=" << diff
                      << " tol=" << tol << std::endl;
            return false;
        }
    }

    return true;
}

int main() {
    const std::vector<Shape> shapes = {
        {3, 3, 3},
        {2, 3, 4},
        {4, 5, 2}
    };

    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr float rtol = 1e-4f;
    constexpr float atol = 1e-5f;

    bool passed = true;

    for (const auto& shape : shapes) {
        int M = shape.M;
        int K = shape.K;
        int N = shape.N;
        size_t size_a = static_cast<size_t>(M) * static_cast<size_t>(K);
        size_t size_b = static_cast<size_t>(K) * static_cast<size_t>(N);
        size_t size_c = static_cast<size_t>(M) * static_cast<size_t>(N);

        std::vector<float> A(size_a);
        std::vector<float> B(size_b);
        for (size_t i = 0; i < size_a; ++i) {
            A[i] = dist(gen);
        }
        for (size_t i = 0; i < size_b; ++i) {
            B[i] = dist(gen);
        }

        std::vector<float> C_naive(size_c, 0.0f);
        hwml::gemm_naive(A.data(), B.data(), C_naive.data(), M, K, N);

        std::vector<float> C_tiled(size_c, 0.0f);
        hwml::gemm_tiled(A.data(), B.data(), C_tiled.data(), M, K, N);
        if (!compare_outputs(C_naive, C_tiled, rtol, atol)) {
            std::cout << "FAIL: gemm_tiled (M=" << M << ", K=" << K << ", N=" << N << ")" << std::endl;
            passed = false;
        }

        std::vector<float> C_neon(size_c, 0.0f);
        hwml::gemm_neon(A.data(), B.data(), C_neon.data(), M, K, N);
        if (!compare_outputs(C_naive, C_neon, rtol, atol)) {
            std::cout << "FAIL: gemm_neon (M=" << M << ", K=" << K << ", N=" << N << ")" << std::endl;
            passed = false;
        }

        std::vector<float> C_mt(size_c, 0.0f);
        hwml::gemm_mt(A.data(), B.data(), C_mt.data(), M, K, N);
        if (!compare_outputs(C_naive, C_mt, rtol, atol)) {
            std::cout << "FAIL: gemm_mt (M=" << M << ", K=" << K << ", N=" << N << ")" << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "PASS: GEMM correctness validation" << std::endl;
    }

    return passed ? 0 : 1;
}