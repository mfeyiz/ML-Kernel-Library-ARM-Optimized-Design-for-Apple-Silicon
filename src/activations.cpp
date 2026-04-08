#include "activations.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace hwml {

static inline float32x4_t exp_ps(float32x4_t x) {
    const float32x4_t ln2 = vdupq_n_f32(0.69314718056f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    
    float32x4_t abs_x = vabsq_f32(x);
    
    float32x4_t k_f = vmulq_f32(abs_x, vrecpeq_f32(ln2));
    int32x4_t k_i = vcvtq_s32_f32(k_f);
    k_f = vcvtq_f32_s32(k_i);
    
    uint32x4_t exp_bits = vreinterpretq_u32_s32(vshlq_n_s32(vaddq_s32(k_i, vdupq_n_s32(127)), 23));
    float32x4_t two_k = vreinterpretq_f32_u32(exp_bits);
    
    float32x4_t r = vsubq_f32(abs_x, vmulq_f32(k_f, ln2));
    float32x4_t xx = vmulq_f32(r, r);
    
    float32x4_t p = vdupq_n_f32(1.0f);
    p = vmlaq_f32(p, xx, vdupq_n_f32(0.008301010459f));
    p = vmlaq_f32(p, xx, vdupq_n_f32(0.0419867050f));
    p = vmlaq_f32(p, xx, vdupq_n_f32(0.16765347345f));
    p = vmlaq_f32(p, xx, vdupq_n_f32(0.99923393535f));
    
    float32x4_t y = vaddq_f32(p, vmulq_f32(xx, vmulq_f32(p, p)));
    float32x4_t result = vmulq_f32(y, two_k);
    
    uint32x4_t mask = vcgtq_f32(x, vdupq_n_f32(0.0f));
    result = vbslq_f32(mask, result, vrecpeq_f32(result));
    
    return result;
}

static inline float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void relu_neon(float* X, int size) {
    constexpr int UNROLL = 4;
    int i = 0;
    
    for (; i + 16 <= size; i += 16) {
        for (int u = 0; u < UNROLL; ++u) {
            float32x4_t vec = vld1q_f32(&X[i + u * 4]);
            float32x4_t zero = vdupq_n_f32(0.0f);
            float32x4_t result = vmaxq_f32(vec, zero);
            vst1q_f32(&X[i + u * 4], result);
        }
    }
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t vec = vld1q_f32(&X[i]);
        float32x4_t zero = vdupq_n_f32(0.0f);
        float32x4_t result = vmaxq_f32(vec, zero);
        vst1q_f32(&X[i], result);
    }
    
    for (; i < size; ++i) {
        X[i] = std::max(0.0f, X[i]);
    }
}

void sigmoid_neon(float* X, int size) {
    constexpr int UNROLL = 4;
    int i = 0;
    
    for (; i + 16 <= size; i += 16) {
        for (int u = 0; u < UNROLL; ++u) {
            float32x4_t x = vld1q_f32(&X[i + u * 4]);
            float32x4_t neg_x = vnegq_f32(x);
            float32x4_t exp_neg_x = exp_ps(neg_x);
            float32x4_t one = vdupq_n_f32(1.0f);
            float32x4_t denom = vaddq_f32(exp_neg_x, one);
            float32x4_t result = vrecpeq_f32(denom);
            vst1q_f32(&X[i + u * 4], result);
        }
    }
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(&X[i]);
        float32x4_t neg_x = vnegq_f32(x);
        float32x4_t exp_neg_x = exp_ps(neg_x);
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t denom = vaddq_f32(exp_neg_x, one);
        float32x4_t result = vrecpeq_f32(denom);
        vst1q_f32(&X[i], result);
    }
    
    for (; i < size; ++i) {
        X[i] = sigmoid_scalar(X[i]);
    }
}

void relu_mt(float* X, int size) {
    unsigned int max_threads = std::thread::hardware_concurrency();
    if (max_threads == 0) max_threads = 1;
    
    unsigned int num_threads = std::min(max_threads, 8u);
    
    if (size < 10000 || num_threads == 1) {
        relu_neon(X, size);
        return;
    }
    
    std::vector<std::thread> threads;
    
    int chunk_size = size / num_threads;
    int remainder = size % num_threads;
    
    int current = 0;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start = current;
        int end = start + chunk_size + (t < remainder ? 1 : 0);
        current = end;
        
        int start_copy = start;
        int end_copy = end;
        
        threads.push_back(std::thread([=]() {
            relu_neon(X + start_copy, end_copy - start_copy);
        }));
    }
    
    for (auto& th : threads) {
        th.join();
    }
}

void sigmoid_mt(float* X, int size) {
    unsigned int max_threads = std::thread::hardware_concurrency();
    if (max_threads == 0) max_threads = 1;
    
    unsigned int num_threads = std::min(max_threads, 8u);
    
    if (size < 10000 || num_threads == 1) {
        sigmoid_neon(X, size);
        return;
    }
    
    std::vector<std::thread> threads;
    
    int chunk_size = size / num_threads;
    int remainder = size % num_threads;
    
    int current = 0;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int start = current;
        int end = start + chunk_size + (t < remainder ? 1 : 0);
        current = end;
        
        int start_copy = start;
        int end_copy = end;
        
        threads.push_back(std::thread([=]() {
            sigmoid_neon(X + start_copy, end_copy - start_copy);
        }));
    }
    
    for (auto& th : threads) {
        th.join();
    }
}

} // namespace hwml