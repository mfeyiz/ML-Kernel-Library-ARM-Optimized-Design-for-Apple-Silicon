#include "gemm.h"

#if __has_include(<Accelerate/Accelerate.h>)
#include <Accelerate/Accelerate.h>
#else
#error "Accelerate framework headers not found"
#endif

#include <limits>
#include <stdexcept>

namespace hwml {

void gemm_accelerate(const float* A,
                     const float* B,
                     float* C,
                     size_t M,
                     size_t K,
                     size_t N,
                     float alpha,
                     float beta) {
    if (!A || !B || !C) {
        throw std::invalid_argument("gemm_accelerate: null pointer");
    }

    if (M > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        K > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("gemm_accelerate: dimensions exceed int range");
    }

    // Row-major, A(MxK) * B(KxN) = C(MxN)
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<int>(M),
                static_cast<int>(N),
                static_cast<int>(K),
                alpha,
                A,
                static_cast<int>(K),
                B,
                static_cast<int>(N),
                beta,
                C,
                static_cast<int>(N));
}

} // namespace hwml
