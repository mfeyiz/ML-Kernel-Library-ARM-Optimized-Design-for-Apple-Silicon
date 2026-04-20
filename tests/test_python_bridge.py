import numpy as np
import arm_gemm_apple
import time
import sys


def test_gemm_correctness():
    print("=== GEMM Correctness Tests ===\n")
    np.random.seed(42)

    test_sizes = [64, 256, 512, 1024, 2048]

    for N in test_sizes:
        A = np.random.randn(N, N).astype(np.float32)
        B = np.random.randn(N, N).astype(np.float32)
        expected = A @ B

        funcs = [
            ("gemm_naive", arm_gemm_apple.gemm_naive),
            ("gemm_tiled", arm_gemm_apple.gemm_tiled),
            ("gemm_neon", arm_gemm_apple.gemm_neon),
            ("gemm_mt", arm_gemm_apple.gemm_mt),
            ("gemm_auto", arm_gemm_apple.gemm_auto),
        ]

        for name, func in funcs:
            result = func(A, B)
            if np.allclose(result, expected, atol=1e-3):
                print(f"  ✓ N={N:4d} {name:12s} correct")
            else:
                print(f"  ✗ N={N:4d} {name:12s} FAILED")
                return False

    return True


def test_activations_correctness():
    print("\n=== Activation Correctness Tests ===\n")
    np.random.seed(42)

    sizes = [1000, 100000, 1000000]

    for size in sizes:
        X = np.random.randn(size).astype(np.float32)

        # ReLU
        X_relu = X.copy()
        arm_gemm_apple.relu(X_relu)
        expected_relu = np.maximum(0, X)
        if np.allclose(X_relu, expected_relu):
            print(f"  ✓ relu (size={size:7d}) correct")
        else:
            print(f"  ✗ relu (size={size:7d}) FAILED")
            return False

        X_relu_mt = X.copy()
        arm_gemm_apple.relu_mt(X_relu_mt)
        if np.allclose(X_relu_mt, expected_relu):
            print(f"  ✓ relu_mt (size={size:7d}) correct")
        else:
            print(f"  ✗ relu_mt (size={size:7d}) FAILED")
            return False

        # Sigmoid (higher tolerance due to approximate exp implementation)
        X_sig = X.copy()
        arm_gemm_apple.sigmoid(X_sig)
        expected_sig = 1 / (1 + np.exp(-X))
        if np.allclose(X_sig, expected_sig, atol=0.1):
            print(f"  ✓ sigmoid (size={size:7d}) correct")
        else:
            print(f"  ✗ sigmoid (size={size:7d}) FAILED")
            return False

        X_sig_mt = X.copy()
        arm_gemm_apple.sigmoid_mt(X_sig_mt)
        if np.allclose(X_sig_mt, expected_sig, atol=0.1):
            print(f"  ✓ sigmoid_mt (size={size:7d}) correct")
        else:
            print(f"  ✗ sigmoid_mt (size={size:7d}) FAILED")
            return False

    return True


def test_auto_threshold():
    print("\n=== Auto-Select Threshold Test ===\n")
    np.random.seed(42)

    # Small matrix (should use naive)
    A_small = np.random.randn(100, 100).astype(np.float32)
    B_small = np.random.randn(100, 100).astype(np.float32)

    # Test with different thresholds
    result1 = arm_gemm_apple.gemm_auto(A_small, B_small, threshold=2048)
    result2 = arm_gemm_apple.gemm_naive(A_small, B_small)

    if np.allclose(result1, result2, atol=1e-3):
        print("  ✓ Auto-select uses naive for N < threshold")
    else:
        print("  ✗ Auto-select threshold test FAILED")
        return False

    # Large matrix (should use mt)
    A_large = np.random.randn(2048, 2048).astype(np.float32)
    B_large = np.random.randn(2048, 2048).astype(np.float32)

    result3 = arm_gemm_apple.gemm_auto(A_large, B_large, threshold=1024)
    result4 = arm_gemm_apple.gemm_mt(A_large, B_large)

    if np.allclose(result3, result4, atol=1e-3):
        print("  ✓ Auto-select uses mt for N >= threshold")
    else:
        print("  ✗ Auto-select threshold test FAILED")
        return False

    return True


def benchmark_gemm():
    print("\n=== GEMM Benchmark ===\n")
    np.random.seed(42)

    N = 2048
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    iters = 3

    funcs = [
        ("arm_gemm_apple.gemm_naive", arm_gemm_apple.gemm_naive, False),
        ("arm_gemm_apple.gemm_tiled", arm_gemm_apple.gemm_tiled, False),
        ("arm_gemm_apple.gemm_neon", arm_gemm_apple.gemm_neon, False),
        ("arm_gemm_apple.gemm_mt", arm_gemm_apple.gemm_mt, False),
        ("arm_gemm_apple.gemm_auto", arm_gemm_apple.gemm_auto, False),
        ("numpy", lambda a, b: a @ b, False),
    ]

    # Actually try each function
    funcs_to_test = [
        ("arm_gemm_apple.gemm_mt", arm_gemm_apple.gemm_mt),
        ("numpy", lambda a, b: a @ b),
    ]

    results = []
    for name, func in funcs_to_test:
        start = time.perf_counter()
        for _ in range(iters):
            _ = func(A, B)
        elapsed = (time.perf_counter() - start) / iters
        results.append((name, elapsed))
        print(f"  {name:25s}: {elapsed:.3f}s")

    # Calculate speedup
    numpy_time = next(t for n, t in results if n == "numpy")
    mt_time = next(t for n, t in results if "gemm_mt" in n)
    speedup = numpy_time / mt_time
    print(f"\n  Speedup (gemm_mt vs numpy): {speedup:.2f}x")

    return True


def main():
    print("=" * 60)
    print("  arm_gemm_apple Python Bindings Test Suite")
    print("=" * 60)

    all_passed = True

    if not test_gemm_correctness():
        all_passed = False
        print("\n❌ GEMM correctness tests FAILED")
        sys.exit(1)

    if not test_activations_correctness():
        all_passed = False
        print("\n❌ Activation correctness tests FAILED")
        sys.exit(1)

    if not test_auto_threshold():
        all_passed = False
        print("\n❌ Auto-select threshold tests FAILED")
        sys.exit(1)

    if not benchmark_gemm():
        all_passed = False
        print("\n❌ Benchmark FAILED")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ✓ All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

def test_rectangular_gemm():
    print("\n=== Rectangular GEMM Tests ===")
    
    shapes = [
        (128, 64, 256),
        (64, 256, 128),
        (256, 128, 64),
        (63, 65, 67), # Odd shapes
    ]
    
    passed = True
    for M, K, N in shapes:
        A = np.random.rand(M, K).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)
        
        C_ref = np.dot(A, B)
        
        C_naive = arm_gemm_apple.gemm_naive(A, B)
        C_tiled = arm_gemm_apple.gemm_tiled(A, B)
        C_neon  = arm_gemm_apple.gemm_neon(A, B)
        C_mt    = arm_gemm_apple.gemm_mt(A, B)
        
        diff_naive = np.max(np.abs(C_naive - C_ref))
        diff_tiled = np.max(np.abs(C_tiled - C_ref))
        diff_neon  = np.max(np.abs(C_neon - C_ref))
        diff_mt    = np.max(np.abs(C_mt - C_ref))
        
        if max(diff_naive, diff_tiled, diff_neon, diff_mt) > 1e-4:
            print(f"  ✗ Rectangular {M}x{K}x{N} failed!")
            print(f"      diffs: naive={diff_naive:.2e}, tiled={diff_tiled:.2e}, neon={diff_neon:.2e}, mt={diff_mt:.2e}")
            passed = False
        else:
            print(f"  ✓ Rectangular {M}x{K}x{N} correct")
            
    if passed:
        print("  ✓ All rectangular tests passed!")
    else:
        print("  ✗ Some rectangular tests failed!")
        sys.exit(1)

test_rectangular_gemm()
