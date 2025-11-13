"""
Example: Using exact optimal transport (EMD) solver.
"""

import numpy as np
import matplotlib.pyplot as plt
from otsb import emd, wasserstein_distance
from otsb.utils import squared_euclidean_cost, plot_transport_plan


def main():
    # Set random seed
    np.random.seed(42)

    # Create 1D distributions
    n_source = 15
    n_target = 15

    # Source: uniform on [0, 1]
    X = np.linspace(0, 1, n_source)

    # Target: two clusters
    Y = np.concatenate([np.linspace(0.2, 0.4, n_target // 2),
                        np.linspace(0.6, 0.8, n_target // 2)])

    # Uniform weights
    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target

    # Compute cost matrix (squared Euclidean for Wasserstein-2)
    C = squared_euclidean_cost(X[:, None], Y[:, None])

    # Solve exact OT
    print("Solving exact optimal transport...")
    P, result = emd(a, b, C)

    print(f"Success: {result['success']}")
    print(f"Optimal cost: {result['cost']:.4f}")

    # Compute Wasserstein distance
    W2 = wasserstein_distance(a, b, X[:, None], Y[:, None], p=2)
    print(f"Wasserstein-2 distance: {W2:.4f}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot transport plan matrix
    im = axes[0].imshow(P, cmap="viridis", aspect="auto")
    axes[0].set_xlabel("Target index")
    axes[0].set_ylabel("Source index")
    axes[0].set_title("Exact OT Transport Plan")
    plt.colorbar(im, ax=axes[0])

    # Plot 1D transport plan
    plot_transport_plan(X, Y, P, ax=axes[1], threshold=1e-6)

    plt.tight_layout()
    plt.savefig("exact_ot_example.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to exact_ot_example.png")
    plt.show()


if __name__ == "__main__":
    main()
