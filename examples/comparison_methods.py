"""
Example: Comparing Different OT Methods
This example can run directly after cloning without installation.
"""

import sys
import os
# Add parent directory to path to import otsb without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import time
from otsb import sinkhorn, sinkhorn_log, emd
from otsb.utils import squared_euclidean_cost


def main():
    print("=" * 60)
    print("Comparing OT Methods: Sinkhorn vs Log-Sinkhorn vs Exact")
    print("=" * 60)

    # Set random seed
    np.random.seed(42)

    # Create 2D distributions
    n, m = 50, 50

    # Source: Gaussian mixture
    X1 = np.random.randn(n // 2, 2) * 0.3 + np.array([-1, -1])
    X2 = np.random.randn(n // 2, 2) * 0.3 + np.array([1, 1])
    X = np.vstack([X1, X2])

    # Target: Another Gaussian mixture
    Y1 = np.random.randn(m // 2, 2) * 0.3 + np.array([1, -1])
    Y2 = np.random.randn(m // 2, 2) * 0.3 + np.array([-1, 1])
    Y = np.vstack([Y1, Y2])

    # Uniform weights
    a = np.ones(n) / n
    b = np.ones(m) / m

    # Compute cost matrix
    C = squared_euclidean_cost(X, Y)

    print(f"\nProblem size: {n} × {m}")
    print(f"Cost matrix range: [{C.min():.4f}, {C.max():.4f}]")

    # Test different regularization parameters for Sinkhorn
    reg_values = [0.01, 0.05, 0.1, 0.5]

    print("\n" + "-" * 60)
    print("Testing Sinkhorn with different regularization parameters:")
    print("-" * 60)

    sinkhorn_results = []
    for reg in reg_values:
        start = time.time()
        P, log_dict = sinkhorn(a, b, C, reg=reg, log=True, tol=1e-8)
        elapsed = time.time() - start
        cost = np.sum(P * C)

        sinkhorn_results.append({
            'reg': reg,
            'cost': cost,
            'time': elapsed,
            'iters': log_dict['num_iter'],
            'P': P
        })

        print(f"  reg={reg:5.2f}: cost={cost:8.4f}, "
              f"iters={log_dict['num_iter']:4d}, time={elapsed:.4f}s")

    # Test log-domain Sinkhorn with small regularization
    print("\n" + "-" * 60)
    print("Testing Log-Sinkhorn (numerically stable):")
    print("-" * 60)

    start = time.time()
    P_log, log_dict_log = sinkhorn_log(a, b, C, reg=0.01, log=True, tol=1e-8)
    elapsed_log = time.time() - start
    cost_log = np.sum(P_log * C)

    print(f"  reg=0.01: cost={cost_log:8.4f}, "
          f"iters={log_dict_log['num_iter']:4d}, time={elapsed_log:.4f}s")

    # Test exact OT
    print("\n" + "-" * 60)
    print("Testing Exact OT (Linear Programming):")
    print("-" * 60)

    start = time.time()
    P_exact, result_exact = emd(a, b, C)
    elapsed_exact = time.time() - start

    print(f"  Exact cost: {result_exact['cost']:.4f}, time={elapsed_exact:.4f}s")

    # Visualize results
    fig = plt.figure(figsize=(16, 10))

    # Plot source and target
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(X[:, 0], X[:, 1], c='blue', s=50, alpha=0.6, label='Source')
    ax1.scatter(Y[:, 0], Y[:, 1], c='red', s=50, alpha=0.6, label='Target')
    ax1.set_title('Source and Target Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot Sinkhorn results for different reg values
    for idx, result in enumerate(sinkhorn_results[:4]):
        ax = plt.subplot(2, 3, idx + 2)
        im = ax.imshow(result['P'], cmap='viridis', aspect='auto')
        ax.set_title(f'Sinkhorn (reg={result["reg"]:.2f})\n'
                     f'cost={result["cost"]:.4f}, t={result["time"]:.3f}s')
        ax.set_xlabel('Target index')
        ax.set_ylabel('Source index')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Plot exact OT
    ax6 = plt.subplot(2, 3, 6)
    im = ax6.imshow(P_exact, cmap='viridis', aspect='auto')
    ax6.set_title(f'Exact OT\ncost={result_exact["cost"]:.4f}, t={elapsed_exact:.3f}s')
    ax6.set_xlabel('Target index')
    ax6.set_ylabel('Source index')
    plt.colorbar(im, ax=ax6, fraction=0.046)

    plt.tight_layout()
    output_file = 'comparison_ot_methods.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_file}")

    # Create comparison plot
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cost vs regularization
    costs = [r['cost'] for r in sinkhorn_results]
    regs = [r['reg'] for r in sinkhorn_results]
    times = [r['time'] for r in sinkhorn_results]

    axes[0].plot(regs, costs, 'o-', linewidth=2, markersize=8, label='Sinkhorn')
    axes[0].axhline(result_exact['cost'], color='red', linestyle='--',
                    linewidth=2, label='Exact OT')
    axes[0].set_xlabel('Regularization parameter (ε)')
    axes[0].set_ylabel('Transport cost')
    axes[0].set_title('Cost vs Regularization')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # Iterations vs regularization
    iters = [r['iters'] for r in sinkhorn_results]
    axes[1].plot(regs, iters, 's-', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Regularization parameter (ε)')
    axes[1].set_ylabel('Number of iterations')
    axes[1].set_title('Convergence Speed vs Regularization')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    plt.tight_layout()
    output_file2 = 'comparison_analysis.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved analysis to {output_file2}")

    plt.show()

    print("\n" + "=" * 60)
    print("Key Insights:")
    print("-" * 60)
    print(f"• Smaller regularization → closer to exact OT (but slower)")
    print(f"• Exact OT cost: {result_exact['cost']:.4f}")
    print(f"• Best Sinkhorn cost (reg=0.01): {sinkhorn_results[0]['cost']:.4f}")
    print(f"• Difference: {abs(sinkhorn_results[0]['cost'] - result_exact['cost']):.4f}")
    print(f"• Sinkhorn is {elapsed_exact/sinkhorn_results[2]['time']:.1f}x faster (reg=0.1)")
    print("=" * 60)


if __name__ == "__main__":
    main()
