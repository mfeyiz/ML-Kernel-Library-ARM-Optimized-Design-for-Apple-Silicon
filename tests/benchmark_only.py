import numpy as np
import arm_gemm_apple
import time
import sys
import os
import subprocess


def benchmark_gemm():
    print("\n=== GEMM Benchmark ===\n")
    np.random.seed(42)

    N = 2048
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    iters = 3

    funcs_to_test = [
        ("gemm_naive", arm_gemm_apple.gemm_naive),
        ("gemm_tiled", arm_gemm_apple.gemm_tiled),
        ("gemm_neon", arm_gemm_apple.gemm_neon),
        ("gemm_mt", arm_gemm_apple.gemm_mt),
        ("gemm_auto", arm_gemm_apple.gemm_auto),
    ]

    if hasattr(arm_gemm_apple, "gemm_accelerate"):
        funcs_to_test.append(("gemm_accelerate", arm_gemm_apple.gemm_accelerate))

    funcs_to_test.append(("numpy", lambda a, b: a @ b))

    results = []
    for name, func in funcs_to_test:
        start = time.perf_counter()
        for _ in range(iters):
            _ = func(A, B)
        elapsed = (time.perf_counter() - start) / iters
        results.append((name, elapsed))
        print(f"  {name:15s}: {elapsed:.3f}s")

    numpy_time = next(t for n, t in results if n == "numpy")
    print(f"\n  NumPy baseline: {numpy_time:.3f}s")

    with open("benchmark_results.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  arm_gemm_apple GEMM Benchmark Results\n")
        f.write("  Matrix size: 2048 x 2048 (float32)\n")
        f.write("  Iterations: 3\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"{'Function':<20} {'Time (s)':<15} {'Speedup vs NumPy':<20}\n")
        f.write("-" * 55 + "\n")

        for name, elapsed in results:
            speedup = numpy_time / elapsed if elapsed > 0 else 0
            f.write(f"{name:<20} {elapsed:<15.3f} {speedup:<20.2f}x\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"NumPy is using Apple's Accelerate framework (BLAS)\n")
        f.write("which is highly optimized for Apple Silicon.\n")
        f.write("=" * 60 + "\n")

        f.write("\n\n" + "=" * 60 + "\n")
        f.write("  ML Demo Output (tests/ml_demo.py)\n")
        f.write("=" * 60 + "\n")

        env = os.environ.copy()
        # Preserve / ensure PYTHONPATH points to the build dir so arm_gemm_apple can be imported.
        if "PYTHONPATH" not in env:
            env["PYTHONPATH"] = "build"
        elif "build" not in env["PYTHONPATH"].split(":"):
            env["PYTHONPATH"] = "build:" + env["PYTHONPATH"]

        proc = subprocess.run(
            [sys.executable, os.path.join("tests", "ml_demo.py")],
            capture_output=True,
            text=True,
            env=env,
        )

        if proc.stdout:
            f.write(proc.stdout)
        if proc.stderr:
            f.write("\n[stderr]\n")
            f.write(proc.stderr)
        f.write("\n" + "=" * 60 + "\n")

    print("\n  Results saved to benchmark_results.txt")
    return True


def main():
    print("=" * 60)
    print("  arm_gemm_apple GEMM Benchmark")
    print("=" * 60)
    benchmark_gemm()


if __name__ == "__main__":
    main()
