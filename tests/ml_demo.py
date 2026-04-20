import numpy as np
import arm_gemm_apple as hwml
import time
import sys


def benchmark_function(func, *args, warmup=1, iterations=3, **kwargs):
    for _ in range(warmup):
        func(*args, **kwargs)

    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) / iterations
    return elapsed, result


def test_square_gemm():
    print("=" * 60)
    print("Square Matrix GEMM Benchmark")
    print("=" * 60)

    np.random.seed(42)

    sizes = [64, 128, 256, 512, 1024]

    print(
        "\n{:>6} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "N", "naive", "tiled", "neon", "mt", "numpy"
        )
    )
    print("-" * 65)

    for N in sizes:
        A = np.random.randn(N, N).astype(np.float32)
        B = np.random.randn(N, N).astype(np.float32)

        t_naive, _ = benchmark_function(hwml.gemm_naive, A, B)
        t_tiled, _ = benchmark_function(hwml.gemm_tiled, A, B)
        t_neon, _ = benchmark_function(hwml.gemm_neon, A, B)
        t_mt, _ = benchmark_function(hwml.gemm_mt, A, B)
        t_np, _ = benchmark_function(lambda a, b: a @ b, A, B)

        print(
            "{:>6} {:>9.3f}s {:>9.3f}s {:>9.3f}s {:>9.3f}s {:>9.3f}s".format(
                N, t_naive, t_tiled, t_neon, t_mt, t_np
            )
        )

    return True


def test_activation_functions():
    print("\n" + "=" * 60)
    print("Activation Functions Benchmark")
    print("=" * 60)

    np.random.seed(42)

    sizes = [1000, 10000, 100000, 1000000, 10000000]

    print(
        "\n{:>10} {:>12} {:>12} {:>12}".format("Size", "ReLU", "Sigmoid", "NumPy ReLU")
    )
    print("-" * 50)

    for size in sizes:
        X = np.random.randn(size).astype(np.float32)

        X_relu = X.copy()
        t_relu, _ = benchmark_function(hwml.relu, X_relu)

        X_sig = X.copy()
        t_sig, _ = benchmark_function(hwml.sigmoid, X_sig)

        t_np, _ = benchmark_function(lambda x: np.maximum(0, x), X)

        print(
            "{:>10} {:>11.4f}s {:>11.4f}s {:>11.4f}s".format(size, t_relu, t_sig, t_np)
        )

    return True


def test_alpha_beta():
    print("\n" + "=" * 60)
    print("Alpha/Beta Parameter Test")
    print("=" * 60)

    np.random.seed(42)

    N = 512
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    C = np.random.randn(N, N).astype(np.float32)

    print("\nTesting C = alpha * A @ B + beta * C")

    alpha = 0.5
    beta = 0.3

    result = hwml.gemm(A, B, alpha=alpha, beta=beta)
    expected = alpha * (A @ B) + beta * C

    if np.allclose(result, expected, atol=1e-4):
        print("✓ Alpha/beta computation correct!")
    else:
        print("✗ Alpha/beta computation FAILED")
        return False

    return True


def test_gemm_with_bias():
    print("\n" + "=" * 60)
    print("GEMM + Bias (Linear Layer)")
    print("=" * 60)

    np.random.seed(42)

    batch_size = 256
    input_dim = 512
    output_dim = 10

    X = np.random.randn(batch_size, input_dim).astype(np.float32)
    W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01
    b = np.zeros(output_dim, dtype=np.float32)

    print(f"\nLinear layer: {input_dim} -> {output_dim}")
    print(f"Batch size: {batch_size}")

    time1, Z = benchmark_function(hwml.gemm, X, W)
    print(f"\nGEMM (X @ W):  {time1 * 1000:.2f} ms")

    for i in range(batch_size):
        Z[i] += b

    time2, A = benchmark_function(hwml.relu, Z.copy())
    print(f"ReLU:           {time2 * 1000:.2f} ms")

    total = time1 + time2
    print(f"Total:          {total * 1000:.2f} ms")

    return True


def test_batch_inference():
    print("\n" + "=" * 60)
    print("Batch Inference (Multiple Forward Passes)")
    print("=" * 60)

    np.random.seed(42)

    batch_sizes = [1, 8, 32, 128, 256]
    input_dim = 512
    hidden_dim = 256
    output_dim = 10

    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.01
    b2 = np.zeros(output_dim, dtype=np.float32)

    print(f"\nNetwork: {input_dim} -> {hidden_dim} -> {output_dim}")

    print(
        "\n{:>12} {:>12} {:>12} {:>12}".format(
            "Batch", "Forward", "Per-sample", "Throughput"
        )
    )
    print("-" * 52)

    for batch_size in batch_sizes:
        X = np.random.randn(batch_size, input_dim).astype(np.float32)

        time1, Z1 = benchmark_function(hwml.gemm, X, W1)
        Z1 = Z1 + b1
        time2, A1 = benchmark_function(hwml.relu, Z1.copy())

        time3, Z2 = benchmark_function(hwml.gemm, A1, W2)
        Z2 = Z2 + b2
        time4, A2 = benchmark_function(hwml.sigmoid, Z2.copy())

        total = time1 + time2 + time3 + time4
        per_sample = total / batch_size * 1000
        throughput = batch_size / total

        print(
            "{:>12} {:>11.2f}ms {:>11.2f}ms {:>11.0f} samples/s".format(
                batch_size, total * 1000, per_sample, throughput
            )
        )

    return True


def main():
    print("\n" + "=" * 60)
    print("  arm_gemm_apple ML Benchmark Suite")
    print("  Apple Silicon Optimized GEMM Library")
    print("=" * 60)

    tests = [
        ("Square GEMM", test_square_gemm),
        ("Activations", test_activation_functions),
        ("Alpha/Beta", test_alpha_beta),
        ("GEMM + Bias", test_gemm_with_bias),
        ("Batch Inference", test_batch_inference),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {name}: PASSED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name}: FAILED - {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
