cmake_minimum_required(VERSION 3.27)
project(HWMLKernels VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_INIT "-O3 -mcpu=apple-m1")
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=apple-m1")

# Apple Accelerate framework (BLAS)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(pybind11 REQUIRED)

add_library(hwml_kernels STATIC
    src/gemm_naive.cpp
    src/gemm_tiled.cpp
    src/gemm_neon.cpp
    src/gemm_mt.cpp
    src/gemm_accelerate.cpp
    src/gemm_fp16.cpp
    src/cache_utils.cpp
    src/activations.cpp
)

target_link_libraries(hwml_kernels PUBLIC ${ACCELERATE_FRAMEWORK})

target_include_directories(hwml_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(hwml_kernels PRIVATE
    -std=c++17
    -O3
    -mcpu=apple-m1
    -flax-vector-conversions=all
)

# Python module
pybind11_add_module(arm_gemm_apple
    src/bindings.cpp
    src/gemm_naive.cpp
    src/gemm_tiled.cpp
    src/gemm_neon.cpp
    src/gemm_mt.cpp
    src/gemm_accelerate.cpp
    src/gemm_fp16.cpp
    src/cache_utils.cpp
    src/activations.cpp
)

target_link_libraries(arm_gemm_apple PRIVATE ${ACCELERATE_FRAMEWORK})

target_include_directories(arm_gemm_apple PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(arm_gemm_apple PRIVATE
    -std=c++17
    -O3
    -mcpu=apple-m1
)

add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE hwml_kernels)
target_compile_options(test_gemm PRIVATE -std=c++17 -O3 -mcpu=apple-m1)

add_executable(bench_gemm benchmarks/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE hwml_kernels)
target_compile_options(bench_gemm PRIVATE -std=c++17 -O3 -mcpu=apple-m1)

add_executable(test_activations tests/test_activations.cpp)
target_link_libraries(test_activations PRIVATE hwml_kernels)
target_compile_options(test_activations PRIVATE -std=c++17 -O3 -mcpu=apple-m1)

add_executable(bench_activations benchmarks/bench_activations.cpp)
target_link_libraries(bench_activations PRIVATE hwml_kernels)
target_compile_options(bench_activations PRIVATE -std=c++17 -O3 -mcpu=apple-m1)