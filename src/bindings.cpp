#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gemm.h"
#include "activations.h"
#include <algorithm>

namespace py = pybind11;

template <typename T>
using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

static void validate_gemm_input(const py::array& A, const py::array& B) {
    auto bufA = A.request();
    auto bufB = B.request();
    
    if (bufA.ndim < 1 || bufA.ndim > 2 || bufB.ndim < 1 || bufB.ndim > 2) {
        throw std::invalid_argument("A and B must be 1D or 2D arrays");
    }
    
    size_t A_rows = (bufA.ndim == 1) ? 1 : bufA.shape[0];
    size_t A_cols = (bufA.ndim == 1) ? bufA.size : bufA.shape[1];
    size_t B_rows = (bufB.ndim == 1) ? 1 : bufB.shape[0];
    size_t B_cols = (bufB.ndim == 1) ? bufB.size : bufB.shape[1];
    
    if (A_cols != B_rows) {
        throw std::invalid_argument("A.shape[-1] must equal B.shape[0] (matrix dimensions must match)");
    }
    
    if (bufA.itemsize != 4 || bufB.itemsize != 4) {
        throw std::invalid_argument("A and B must be float32 arrays");
    }
}

static void validate_activation_input(py::array& X) {
    auto buf = X.request();
    
    if (buf.ndim > 2) {
        throw std::invalid_argument("X must be 1D or 2D array");
    }
    
    if (buf.itemsize != 4) {
        throw std::invalid_argument("X must be a float32 array");
    }
}

static py::array_t<float> flatten_array(py::array& X) {
    auto buf = X.request();
    size_t total_size = 1;
    for (size_t i = 0; i < buf.ndim; ++i) {
        total_size *= buf.shape[i];
    }
    
    py::array_t<float> result({static_cast<long>(total_size)});
    auto out_buf = result.request();
    
    float* src = static_cast<float*>(buf.ptr);
    float* dst = static_cast<float*>(out_buf.ptr);
    std::copy(src, src + total_size, dst);
    
    return result;
}

static py::array_t<float> reshape_to_2d(py::array& X, size_t rows, size_t cols) {
    py::array_t<float> result({static_cast<long>(rows), static_cast<long>(cols)});
    auto out_buf = result.request();
    
    auto in_buf = X.request();
    float* src = static_cast<float*>(in_buf.ptr);
    float* dst = static_cast<float*>(out_buf.ptr);
    std::copy(src, src + rows * cols, dst);
    
    return result;
}

// ============ GEMM Wrappers ============

Array<float> gemm_naive_wrapper(const Array<float>& A, const Array<float>& B, float alpha = 1.0f, float beta = 0.0f) {
    validate_gemm_input(A, B);
    auto bufA = A.request();
    auto bufB = B.request();
    size_t M = (bufA.ndim == 1) ? 1 : bufA.shape[0];
    size_t K = (bufA.ndim == 1) ? bufA.size : bufA.shape[1];
    size_t N = (bufB.ndim == 1) ? bufB.size : bufB.shape[1];
    
    Array<float> C = py::array_t<float, py::array::c_style | py::array::forcecast>({static_cast<long>(M), static_cast<long>(N)});
    auto bufC = C.request();
    std::fill_n(static_cast<float*>(bufC.ptr), M * N, 0.0f);
    
    hwml::gemm_naive(static_cast<const float*>(bufA.ptr),
                     static_cast<const float*>(bufB.ptr),
                     static_cast<float*>(bufC.ptr), M, K, N, alpha, beta);
    return C;
}

Array<float> gemm_tiled_wrapper(const Array<float>& A, const Array<float>& B, float alpha = 1.0f, float beta = 0.0f) {
    validate_gemm_input(A, B);
    auto bufA = A.request();
    auto bufB = B.request();
    size_t M = (bufA.ndim == 1) ? 1 : bufA.shape[0];
    size_t K = (bufA.ndim == 1) ? bufA.size : bufA.shape[1];
    size_t N = (bufB.ndim == 1) ? bufB.size : bufB.shape[1];
    
    Array<float> C = py::array_t<float, py::array::c_style | py::array::forcecast>({static_cast<long>(M), static_cast<long>(N)});
    auto bufC = C.request();
    std::fill_n(static_cast<float*>(bufC.ptr), M * N, 0.0f);
    
    hwml::gemm_tiled(static_cast<const float*>(bufA.ptr),
                     static_cast<const float*>(bufB.ptr),
                     static_cast<float*>(bufC.ptr), M, K, N, alpha, beta);
    return C;
}

Array<float> gemm_neon_wrapper(const Array<float>& A, const Array<float>& B, float alpha = 1.0f, float beta = 0.0f) {
    validate_gemm_input(A, B);
    auto bufA = A.request();
    auto bufB = B.request();
    size_t M = (bufA.ndim == 1) ? 1 : bufA.shape[0];
    size_t K = (bufA.ndim == 1) ? bufA.size : bufA.shape[1];
    size_t N = (bufB.ndim == 1) ? bufB.size : bufB.shape[1];
    
    Array<float> C = py::array_t<float, py::array::c_style | py::array::forcecast>({static_cast<long>(M), static_cast<long>(N)});
    auto bufC = C.request();
    std::fill_n(static_cast<float*>(bufC.ptr), M * N, 0.0f);
    
    hwml::gemm_neon(static_cast<const float*>(bufA.ptr),
                    static_cast<const float*>(bufB.ptr),
                    static_cast<float*>(bufC.ptr), M, K, N, alpha, beta);
    return C;
}

Array<float> gemm_mt_wrapper(const Array<float>& A, const Array<float>& B, float alpha = 1.0f, float beta = 0.0f) {
    validate_gemm_input(A, B);
    auto bufA = A.request();
    auto bufB = B.request();
    size_t M = (bufA.ndim == 1) ? 1 : bufA.shape[0];
    size_t K = (bufA.ndim == 1) ? bufA.size : bufA.shape[1];
    size_t N = (bufB.ndim == 1) ? bufB.size : bufB.shape[1];
    
    Array<float> C = py::array_t<float, py::array::c_style | py::array::forcecast>({static_cast<long>(M), static_cast<long>(N)});
    auto bufC = C.request();
    std::fill_n(static_cast<float*>(bufC.ptr), M * N, 0.0f);
    
    hwml::gemm_mt(static_cast<const float*>(bufA.ptr),
                  static_cast<const float*>(bufB.ptr),
                  static_cast<float*>(bufC.ptr), M, K, N, alpha, beta);
    return C;
}

Array<float> gemm_auto_wrapper(const Array<float>& A, const Array<float>& B, float alpha = 1.0f, float beta = 0.0f, size_t threshold = 1024) {
    validate_gemm_input(A, B);
    auto bufA = A.request();
    size_t M = bufA.shape[0];
    
    if (M >= threshold) {
        return gemm_mt_wrapper(A, B, alpha, beta);
    } else {
        return gemm_neon_wrapper(A, B, alpha, beta);
    }
}

// ============ Activation Wrappers (in-place) ============

void relu_neon_wrapper(Array<float>& X) {
    validate_activation_input(X);
    auto buf = X.request();
    
    if (buf.ndim == 1) {
        hwml::relu_neon(static_cast<float*>(buf.ptr), buf.size);
    } else {
        size_t total = buf.shape[0] * buf.shape[1];
        hwml::relu_neon(static_cast<float*>(buf.ptr), total);
    }
}

void relu_mt_wrapper(Array<float>& X) {
    validate_activation_input(X);
    auto buf = X.request();
    
    if (buf.ndim == 1) {
        hwml::relu_mt(static_cast<float*>(buf.ptr), buf.size);
    } else {
        size_t total = buf.shape[0] * buf.shape[1];
        hwml::relu_mt(static_cast<float*>(buf.ptr), total);
    }
}

void sigmoid_neon_wrapper(Array<float>& X) {
    validate_activation_input(X);
    auto buf = X.request();
    
    if (buf.ndim == 1) {
        hwml::sigmoid_neon(static_cast<float*>(buf.ptr), buf.size);
    } else {
        size_t total = buf.shape[0] * buf.shape[1];
        hwml::sigmoid_neon(static_cast<float*>(buf.ptr), total);
    }
}

void sigmoid_mt_wrapper(Array<float>& X) {
    validate_activation_input(X);
    auto buf = X.request();
    
    if (buf.ndim == 1) {
        hwml::sigmoid_mt(static_cast<float*>(buf.ptr), buf.size);
    } else {
        size_t total = buf.shape[0] * buf.shape[1];
        hwml::sigmoid_mt(static_cast<float*>(buf.ptr), total);
    }
}

// ============ Module Definition ============

PYBIND11_MODULE(arm_gemm_apple, m) {
    m.doc() = "High-performance Apple Silicon ML kernels (GEMM + Activations)";
    
    // GEMM functions
    m.def("gemm_naive", &gemm_naive_wrapper, "Naive GEMM implementation",
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
    m.def("gemm_tiled", &gemm_tiled_wrapper, "Tiled GEMM implementation",
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
    m.def("gemm_neon", &gemm_neon_wrapper, "NEON vectorized GEMM",
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
    m.def("gemm_mt", &gemm_mt_wrapper, "Multi-threaded GEMM (default for large matrices)",
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);
    m.def("gemm", &gemm_auto_wrapper, "Auto-select GEMM based on matrix size",
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("threshold") = 1024);
    m.def("gemm_auto", &gemm_auto_wrapper, "Auto-select GEMM based on matrix size",
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("threshold") = 1024);
    
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