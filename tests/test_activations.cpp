#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "activations.h"

bool floatEqual(float a, float b, float epsilon = 1e-4f) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0)) return true;
    return std::fabs(a - b) < epsilon;
}

int main() {
    const int SIZE = 8;
    std::vector<float> test_values = {-5.0f, -1.0f, -0.0f, 0.0f, 1.0f, 5.0f, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    
    std::cout << "=== ReLU Tests ===" << std::endl;
    for (int i = 0; i < SIZE; ++i) {
        float x = test_values[i];
        float expected = std::max(0.0f, x);
        float result = x;
        
        hwml::relu_neon(&result, 1);
        
        bool pass = floatEqual(result, expected);
        std::cout << "relu(" << x << ") = " << result << " (expected " << expected << ") " << (pass ? "PASS" : "FAIL") << std::endl;
    }
    
    std::cout << "\n=== Sigmoid Tests ===" << std::endl;
    for (int i = 0; i < SIZE; ++i) {
        float x = test_values[i];
        float expected = 1.0f / (1.0f + std::exp(-x));
        float result = x;
        
        hwml::sigmoid_neon(&result, 1);
        
        bool pass = floatEqual(result, expected, 1e-3f);
        std::cout << "sigmoid(" << x << ") = " << result << " (expected " << expected << ") " << (pass ? "PASS" : "FAIL") << std::endl;
    }
    
    std::cout << "\n=== MT Tests ===" << std::endl;
    std::vector<float> data = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
    std::vector<float> relu_expected = data;
    for (float& x : relu_expected) x = std::max(0.0f, x);
    
    std::vector<float> relu_mt_test = data;
    hwml::relu_mt(relu_mt_test.data(), relu_mt_test.size());
    
    bool relu_mt_pass = true;
    for (int i = 0; i < SIZE; ++i) {
        if (!floatEqual(relu_mt_test[i], relu_expected[i])) {
            relu_mt_pass = false;
            std::cout << "FAIL: relu_mt[" << i << "]" << std::endl;
        }
    }
    std::cout << "relu_mt: " << (relu_mt_pass ? "PASS" : "FAIL") << std::endl;
    
    std::vector<float> sigmoid_expected = data;
    for (float& x : sigmoid_expected) x = 1.0f / (1.0f + std::exp(-x));
    
    std::vector<float> sigmoid_mt_test = data;
    hwml::sigmoid_mt(sigmoid_mt_test.data(), sigmoid_mt_test.size());
    
    bool sigmoid_mt_pass = true;
    for (int i = 0; i < SIZE; ++i) {
        if (!floatEqual(sigmoid_mt_test[i], sigmoid_expected[i], 0.05f)) {
            sigmoid_mt_pass = false;
            std::cout << "FAIL: sigmoid_mt[" << i << "] got " << sigmoid_mt_test[i] << " expected " << sigmoid_expected[i] << std::endl;
        }
    }
    std::cout << "sigmoid_mt: " << (sigmoid_mt_pass ? "PASS" : "FAIL") << std::endl;
    
    return (relu_mt_pass && sigmoid_mt_pass) ? 0 : 1;
}