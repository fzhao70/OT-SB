"""
Benchmark: Compare CPU vs GPU performance for PyTorch OT-SB.
This example demonstrates the speedup from GPU acceleration.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from otsb_torch import sinkhorn, SchrodingerBridgeSolver


def benchmark_sinkhorn(sizes, devices):
    """Benchmark Sinkhorn algorithm on different devices."""
    results = {device: [] for device in devices}

    print("=" * 70)
    print("Benchmarking Sinkhorn Algorithm")
    print("=" * 70)

    for n in sizes:
        print(f"\nProblem size: {n} x {n}")

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                print(f"  {device}: SKIPPED (CUDA not available)")
                results[device].append(None)
                continue

            # Create data
            torch.manual_seed(42)
            X = torch.randn(n, 2, device=device)
            Y = torch.randn(n, 2, device=device) + 2
            a = torch.ones(n, device=device) / n
            b = torch.ones(n, device=device) / n

            from otsb_torch.utils import squared_euclidean_cost
            C = squared_euclidean_cost(X, Y)

            # Warm-up
            _ = sinkhorn(a, b, C, reg=0.1, max_iter=100, device=device)

            # Benchmark
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            P, log_dict = sinkhorn(a, b, C, reg=0.1, max_iter=1000, tol=1e-8, log=True, device=device)

            if device == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start
            results[device].append(elapsed)

            print(f"  {device}: {elapsed:.4f}s ({log_dict['num_iter']} iters)")

    return results


def benchmark_schrodinger_bridge(sizes, devices):
    """Benchmark Schrödinger Bridge on different devices."""
    results = {device: [] for device in devices}

    print("\n" + "=" * 70)
    print("Benchmarking Schrödinger Bridge")
    print("=" * 70)

    for n in sizes:
        print(f"\nProblem size: {n} samples")

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                print(f"  {device}: SKIPPED (CUDA not available)")
                results[device].append(None)
                continue

            # Create data
            torch.manual_seed(42)
            X0 = torch.randn(n, 2, device=device) * 0.3
            X1 = torch.randn(n, 2, device=device) * 0.3 + 3

            # Warm-up
            sb = SchrodingerBridgeSolver(n_steps=20, sigma=0.5, max_iter=10, device=device)
            sb.fit(X0, X1)

            # Benchmark
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            sb = SchrodingerBridgeSolver(n_steps=30, sigma=0.5, max_iter=50, device=device)
            sb.fit(X0, X1)

            if device == 'cuda':
                torch.cuda.synchronize()

            elapsed = time.time() - start
            results[device].append(elapsed)

            print(f"  {device}: {elapsed:.4f}s ({sb.history_['num_iter']} iters)")

    return results


def main():
    print("=" * 70)
    print("GPU vs CPU Performance Benchmark for OT-SB (PyTorch)")
    print("=" * 70)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"\nGPU Available: {torch.cuda.get_device_name(0)}")
        devices = ['cpu', 'cuda']
    else:
        print("\nGPU not available, running CPU-only benchmark")
        devices = ['cpu']

    # Test sizes
    sizes = [50, 100, 200, 500]

    # Run benchmarks
    sinkhorn_results = benchmark_sinkhorn(sizes, devices)
    sb_results = benchmark_schrodinger_bridge(sizes[:3], devices)  # Smaller sizes for SB

    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Sinkhorn results
    for device in devices:
        times = sinkhorn_results[device]
        if None not in times:
            axes[0].plot(sizes, times, 'o-', linewidth=2, markersize=8, label=device.upper())

    axes[0].set_xlabel('Problem Size (n × n)')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Sinkhorn Algorithm Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    # Plot Schrödinger Bridge results
    sb_sizes = sizes[:3]
    for device in devices:
        times = sb_results[device]
        if None not in times:
            axes[1].plot(sb_sizes, times, 's-', linewidth=2, markersize=8, label=device.upper())

    axes[1].set_xlabel('Problem Size (n samples)')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Schrödinger Bridge Performance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')

    plt.tight_layout()
    output_file = "gpu_benchmark.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved benchmark results to {output_file}")
    plt.show()

    # Print speedup summary
    if 'cuda' in devices and torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("Speedup Summary (GPU vs CPU)")
        print("=" * 70)
        print("\nSinkhorn Algorithm:")
        for i, n in enumerate(sizes):
            if sinkhorn_results['cpu'][i] and sinkhorn_results['cuda'][i]:
                speedup = sinkhorn_results['cpu'][i] / sinkhorn_results['cuda'][i]
                print(f"  n={n:4d}: {speedup:.2f}x faster on GPU")

        print("\nSchrödinger Bridge:")
        for i, n in enumerate(sb_sizes):
            if sb_results['cpu'][i] and sb_results['cuda'][i]:
                speedup = sb_results['cpu'][i] / sb_results['cuda'][i]
                print(f"  n={n:4d}: {speedup:.2f}x faster on GPU")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
