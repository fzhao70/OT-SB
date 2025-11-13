"""
Example: Simple 1D Optimal Transport
This example can run directly after cloning without installation.
"""

import sys
import os
# Add parent directory to path to import otsb without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from otsb import sinkhorn, emd
from otsb.utils import squared_euclidean_cost, plot_transport_plan


def main():
    print("=" * 60)
    print("1D Optimal Transport Example")
    print("=" * 60)

    # Set random seed
    np.random.seed(42)

    # Create 1D source distribution (bimodal)
    n_source = 30
    X = np.concatenate([
        np.random.randn(n_source // 2) * 0.3 - 2,
        np.random.randn(n_source // 2) * 0.3 + 2
    ])
    X = np.sort(X)  # Sort for better visualization

    # Create 1D target distribution (unimodal)
    n_target = 30
    Y = np.random.randn(n_target) * 0.5
    Y = np.sort(Y)

    # Uniform weights
    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target

    # Compute cost matrix
    C = squared_euclidean_cost(X[:, None], Y[:, None])

    print(f"\nSource distribution: {n_source} samples")
    print(f"Target distribution: {n_target} samples")
    print(f"Cost matrix shape: {C.shape}")

    # Solve with Sinkhorn
    print("\n1. Solving with Sinkhorn algorithm...")
    P_sinkhorn, log_sinkhorn = sinkhorn(a, b, C, reg=0.05, log=True)
    sinkhorn_cost = np.sum(P_sinkhorn * C)
    print(f"   Converged in {log_sinkhorn['num_iter']} iterations")
    print(f"   Regularized cost: {sinkhorn_cost:.4f}")

    # Solve with exact OT
    print("\n2. Solving with Exact OT (EMD)...")
    P_exact, result_exact = emd(a, b, C)
    print(f"   Exact cost: {result_exact['cost']:.4f}")

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Source and target distributions
    axes[0, 0].hist(X, bins=15, alpha=0.6, label='Source', color='blue', density=True)
    axes[0, 0].hist(Y, bins=15, alpha=0.6, label='Target', color='red', density=True)
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Source and Target Distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Sinkhorn transport plan
    plot_transport_plan(X, Y, P_sinkhorn, ax=axes[0, 1], threshold=1e-3)
    axes[0, 1].set_title(f'Sinkhorn Transport Plan (cost={sinkhorn_cost:.4f})')

    # Plot 3: Exact OT transport plan
    plot_transport_plan(X, Y, P_exact, ax=axes[1, 0], threshold=1e-6)
    axes[1, 0].set_title(f'Exact OT Transport Plan (cost={result_exact["cost"]:.4f})')

    # Plot 4: Comparison of transport plans
    im = axes[1, 1].imshow(
        np.abs(P_sinkhorn - P_exact),
        cmap='Reds',
        aspect='auto'
    )
    axes[1, 1].set_xlabel('Target index')
    axes[1, 1].set_ylabel('Source index')
    axes[1, 1].set_title('Difference: |Sinkhorn - Exact|')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    output_file = 'simple_1d_ot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_file}")
    plt.show()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
