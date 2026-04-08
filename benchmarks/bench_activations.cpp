#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include "activations.h"

int main() {
    const int SIZE = 10000000;
    const double bytes = static_cast<double>(SIZE) * sizeof(float);
    const double gb = bytes / 1e9;
    
    std::vector<float> data(SIZE);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    
    for (float& x : data) x = dist(gen);
    
    std::vector<float> data_copy = data;
    auto start = std::chrono::high_resolution_clock::now();
    hwml::relu_neon(data_copy.data(), SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double bw = gb / (ms / 1000.0);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "ReLU NEON:   " << std::setw(8) << ms << " ms, " << std::setw(7) << bw << " GB/s" << std::endl;
    
    data_copy = data;
    start = std::chrono::high_resolution_clock::now();
    hwml::relu_mt(data_copy.data(), SIZE);
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count();
    bw = gb / (ms / 1000.0);
    
    std::cout << "ReLU MT:     " << std::setw(8) << ms << " ms, " << std::setw(7) << bw << " GB/s" << std::endl;
    
    for (float& x : data) x = dist(gen);
    data_copy = data;
    start = std::chrono::high_resolution_clock::now();
    hwml::sigmoid_neon(data_copy.data(), SIZE);
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count();
    bw = gb / (ms / 1000.0);
    
    std::cout << "Sigmoid NEON:" << std::setw(8) << ms << " ms, " << std::setw(7) << bw << " GB/s" << std::endl;
    
    for (float& x : data) x = dist(gen);
    data_copy = data;
    start = std::chrono::high_resolution_clock::now();
    hwml::sigmoid_mt(data_copy.data(), SIZE);
    end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count();
    bw = gb / (ms / 1000.0);
    
    std::cout << "Sigmoid MT:  " << std::setw(8) << ms << " ms, " << std::setw(7) << bw << " GB/s" << std::endl;
    
    return 0;
}