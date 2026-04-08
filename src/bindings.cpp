#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gemm.h"
#include "activations.h"

namespace py = pybind11;

template <typename T>
using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

// ============ GEMM Wrappers ============

Array<float> gemm_naive_wrapper(const Array<float>& A, const Array<float>& B) {
    auto bufA = A.request();
    auto bufB = B.request();
    int N = bufA.shape[0];
    
    Array<float> C = py::array_t<float, py::array::c_style | py::array::forcecast>({N, N});
    auto bufC = C.request();
    std::fill_n(static_cast<float*>(bufC.ptr), N * N, 0.0f);
    
    hwml::gemm_naive(static_cast<const float*>(bufA.ptr),
                     static_cast<const float*>(bufB.ptr),
                     static_cast<float*>(bufC.ptr), N);
    return C;
}

Array<float> gemm_tiled_wrapper(const Array<float>& A, const Array<float>& B) {
    auto bufA = A.request();
    auto bufB = B.request();
    int N = bufA.shape[0];
    
    Array<float> C = py::array_t<float, py::array::c_style | py::array::forcecast>({N, N});
    auto bufC = C.request();
    std::fill_n(static_cast<float*>(bufC.ptr), N * N, 0.0f);
    
    hwml::gemm_tiled(static_cast<const float*>(bufA.ptr),
                     static_cast<const float*>(bufB.ptr),
                     static_cast<float*>(bufC.ptr), N);
    return C;
}

Array<float> gemm_neon_wrapper(const Array<float>& A, const Array<float>& B) {
    auto bufA = A.request();
    auto bufB = B.request();
    int N = bufA.shape[0];
    
    Array<float> C = py::array_t<float, py::array::c_style | py::array::forcecast>({N, N});
    auto bufC = C.request();
    std::fill_n(static_cast<float*>(bufC.ptr), N * N, 0.0f);
    
    hwml::gemm_neon(static_cast<const float*>(bufA.ptr),
                    static_cast<const float*>(bufB.ptr),
                    static_cast<float*>(bufC.ptr), N);
    return C;
}

Array<float> gemm_mt_wrapper(const Array<float>& A, const Array<float>& B) {
    auto bufA = A.request();
    auto bufB = B.request();
    int N = bufA.shape[0];
    
    Array<float> C = py::array_t<float, py::array::c_style | py::array::forcecast>({N, N});
    auto bufC = C.request();
    std::fill_n(static_cast<float*>(bufC.ptr), N * N, 0.0f);
    
    hwml::gemm_mt(static_cast<const float*>(bufA.ptr),
                  static_cast<const float*>(bufB.ptr),
                  static_cast<float*>(bufC.ptr), N);
    return C;
}

Array<float> gemm_auto_wrapper(const Array<float>& A, const Array<float>& B, int threshold) {
    int N = A.request().shape[0];
    
    if (N < threshold) {
        return gemm_naive_wrapper(A, B);
    } else {
        return gemm_mt_wrapper(A, B);
    }
}

// ============ Activation Wrappers (in-place) ============

void relu_neon_wrapper(Array<float>& X) {
    auto buf = X.request();
    hwml::relu_neon(static_cast<float*>(buf.ptr), buf.size);
}

void relu_mt_wrapper(Array<float>& X) {
    auto buf = X.request();
    hwml::relu_mt(static_cast<float*>(buf.ptr), buf.size);
}

void sigmoid_neon_wrapper(Array<float>& X) {
    auto buf = X.request();
    hwml::sigmoid_neon(static_cast<float*>(buf.ptr), buf.size);
}

void sigmoid_mt_wrapper(Array<float>& X) {
    auto buf = X.request();
    hwml::sigmoid_mt(static_cast<float*>(buf.ptr), buf.size);
}

// ============ Module Definition ============

PYBIND11_MODULE(arm_gemm_apple, m) {
    m.doc() = "High-performance Apple Silicon ML kernels (GEMM + Activations)";
    
    // GEMM functions
    m.def("gemm_naive", &gemm_naive_wrapper, "Naive GEMM implementation",
          py::arg("A"), py::arg("B"));
    m.def("gemm_tiled", &gemm_tiled_wrapper, "Tiled GEMM implementation",
          py::arg("A"), py::arg("B"));
    m.def("gemm_neon", &gemm_neon_wrapper, "NEON vectorized GEMM",
          py::arg("A"), py::arg("B"));
    m.def("gemm_mt", &gemm_mt_wrapper, "Multi-threaded GEMM (default for large matrices)",
          py::arg("A"), py::arg("B"));
    m.def("gemm_auto", &gemm_auto_wrapper, "Auto-select GEMM based on matrix size",
          py::arg("A"), py::arg("B"), py::arg("threshold") = 1024);
    
    // Activation functions (in-place)
    m.def("relu", &relu_neon_wrapper, "ReLU activation (NEON, in-place)",
          py::arg("X"));
    m.def("relu_mt", &relu_mt_wrapper, "ReLU activation (multi-threaded, in-place)",
          py::arg("X"));
    m.def("sigmoid", &sigmoid_neon_wrapper, "Sigmoid activation (NEON, in-place)",
          py::arg("X"));
    m.def("sigmoid_mt", &sigmoid_mt_wrapper, "Sigmoid activation (multi-threaded, in-place)",
          py::arg("X"));
}